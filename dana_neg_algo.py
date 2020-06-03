import math
import random
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Callable, Union, Type, Optional

from negmas import (
    AspirationMixin,
    LinearUtilityFunction,
    PassThroughNegotiator,
    MechanismState,
    ResponseType,
    UtilityFunction,
    AgentWorldInterface,
    outcome_is_valid,
    Outcome,
)
from negmas.events import Notifier, Notification

import functools
from negmas.events import Notifier
from negmas.helpers import instantiate
from scml import SCML2020Agent
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020.components.negotiation import ControllerInfo
from scml.scml2020.services import StepController, SyncController
from run_tournament import run
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from negmas import (AgentMechanismInterface, Breach, Contract, Issue,
                    MechanismState, Negotiator, SAONegotiator, AspirationNegotiator, LinearUtilityFunction,
                    SAOController, AspirationMixin, AgentWorldInterface, SAOSyncController, SAOResponse, SAOState,
                    UtilityValue, MappingUtilityFunction, INVALID_UTILITY)
from scml.scml2020 import SCML2020Agent, PredictionBasedTradingStrategy

from scml.scml2020.world import Failure, QUANTITY, UNIT_PRICE, TIME


""" implementation """


class DanasUtilityFunction1(LinearUtilityFunction):
    def __init__(self, controller, *args, **kwargs):
        self.controller = controller
        if self.controller.is_seller:
            args = (1, 1, 10)
        else:
            args = (1, -1, -10)
        super().__init__(args)

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        return super().eval(offer)


class DanasUtilityFunction(MappingUtilityFunction):
    def __init__(self, controller, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.controller = controller
        if not self.controller.is_seller:
            super().__init__(mapping=lambda outcome: (math.exp(outcome["unit_price"]) - 1.5)
                                        * outcome["quantity"]
                if outcome["unit_price"] > 0.0
                else INVALID_UTILITY,
                             default=INVALID_UTILITY)

            # self.func = MappingUtilityFunction(
            #     mapping=lambda outcome: (math.exp(outcome["unit_price"]) - 1.5)
            #                             * outcome["quantity"]
            #     if outcome["unit_price"] > 0.0
            #     else INVALID_UTILITY
            # )
        else:
            super().__init__(mapping=lambda outcome: 1 - outcome["unit_price"],
                             reserved_value=INVALID_UTILITY,
                             default=INVALID_UTILITY)
            # self.func = MappingUtilityFunction(
            #     mapping=lambda outcome: 1 - outcome["unit_price"],
            #     reserved_value=INVALID_UTILITY,
            # )
        x = 0
    # def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
    #     return self.func.eval(offer)

    # def xml(self, issues):
    #     pass


class DanasUtilityFunction2(UtilityFunction):
    def __init__(self, controller=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self._price_weight = 0.7

    def xml(self, issues):
        pass

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        # return super().eval(offer)
        # get my needs and secured amounts arrays

        _needed, _secured = (
            self.controller.target,
            self.controller.secured,
        )

        if offer is None:
            return self.reserved_value

        # offers for contracts that can never be executed have no utility
        t = offer[TIME]
        if t < self.controller.awi.current_step or t > self.controller.awi.n_steps - 1:
            raise Exception
            # return -1000.0

        # offers that exceed my needs have no utility (that can be improved)
        # q = _needed - (offer[QUANTITY] + _secured)
        # if q < 0:
        #     return -1000.0

        # The utility of any offer is a linear combination of its price and how
        # much it satisfy my needs

        if self.controller.is_seller:
            # return offer[UNIT_PRICE]*offer[QUANTITY] + 1 * offer[TIME]
            return offer[QUANTITY] + 10 * offer[UNIT_PRICE] - 1 * offer[TIME]
        else:
            return offer[QUANTITY] - 10 * offer[UNIT_PRICE] - 1 * offer[TIME] #offer[QUANTITY] / offer[UNIT_PRICE] - 1 * offer[TIME]

        # price = offer[UNIT_PRICE] if self.controller.is_seller else -offer[UNIT_PRICE]
        # return self._price_weight * price + (1 - self._price_weight) * q


"""
improvements:
1) change ufun
"""


class DanasNegotiator(AspirationNegotiator):
    def __init__(
            self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # print("===> Dana's Negotiator!!")


""" implementation of regular StepController for every step """


class DanasController(SAOController, AspirationMixin, Notifier):
    """A controller for managing a set of negotiations about selling or buying (but not both)  starting/ending at some
    specific time-step.

    Args:

        target_quantity: The quantity to be secured
        is_seller:  Is this a seller or a buyer
        parent_name: Name of the parent
        horizon: How many steps in the future to allow negotiations for selling to go for.
        step:  The simulation step that this controller is responsible about
        urange: The range of unit prices used for negotiation
        product: The product that this controller negotiates about
        partners: A list of partners to negotiate with
        negotiator_type: The type of the single negotiator used for all negotiations.
        negotiator_params: The parameters of the negotiator used for all negotiations
        max_retries: How many times can the controller try negotiating with each partner.
        negotiations_concluded_callback: A method to be called with the step of this controller and whether is it a
                                         seller when all negotiations are concluded
        *args: Position arguments passed to the base Controller constructor
        **kwargs: Keyword arguments passed to the base Controller constructor


    Remarks:

        - It uses whatever negotiator type on all of its negotiations and it assumes that the ufun will never change
        - Once it accumulates the required quantity, it ends all remaining negotiations
        - It assumes that all ufuns are identical so there is no need to keep a separate negotiator for each one and it
          instantiates a single negotiator that dynamically changes the AMI but always uses the same ufun.

    """

    def __init__(
        self,
        *args,
        target_quantity: int,
        is_seller: bool,
        step: int,
        urange: Tuple[int, int],
        product: int,
        partners: List[str],
        negotiator_type: SAONegotiator,
        horizon: int,
        awi: AgentWorldInterface,
        parent_name: str,
        negotiations_concluded_callback: Callable[[int, bool], None],
        negotiator_params: Dict[str, Any] = None,
        max_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parent_name = parent_name
        self.awi = awi
        self.horizon = horizon
        self.negotiations_concluded_callback = negotiations_concluded_callback
        self.is_seller = is_seller
        self.target = target_quantity
        self.urange = urange
        self.partners = partners
        self.product = product
        negotiator_params = (negotiator_params if negotiator_params is not None else dict())
        self.secured = 0
        self.ufun = DanasUtilityFunction(controller=self)
        negotiator_params["ufun"] = self.ufun
        self.__negotiator = instantiate(negotiator_type, **negotiator_params)

        self.completed = defaultdict(bool)
        self.step = step
        self.retries: Dict[str, int] = defaultdict(int)
        self.max_retries = max_retries

    def join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        raise Exception
        # joined = super().join(negotiator_id, ami, state, ufun=ufun, role=role)
        # if joined:
        #     self.completed[negotiator_id] = False
        # return joined

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        a = self.__negotiator.propose(state)
        # if state.current_offer is not None:
        #     s = 0
        return a

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> ResponseType:
        if self.secured >= self.target:
            return ResponseType.END_NEGOTIATION
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        a = self.__negotiator.respond(offer=offer, state=state)
        return a

##################################### START TRY
    # def __init__(
    #     self,
    #     max_aspiration=1.0,
    #     aspiration_type="boulware",
    #     dynamic_ufun=True,
    #     randomize_offer=False,
    #     can_propose=True,
    #     assume_normalized=False,
    #     ranking=False,
    #     ufun_max=None,
    #     ufun_min=None,
    #     presort: bool = True,
    #     tolerance: float = 0.01,
    #     **kwargs,
    # ):
    #     self.ordered_outcomes = []
    #     self.ufun_max = ufun_max
    #     self.ufun_min = ufun_min
    #     self.ranking = ranking
    #     self.tolerance = tolerance
    #     if assume_normalized:
    #         self.ufun_max, self.ufun_min = 1.0, 0.0
    #     super().__init__(
    #         assume_normalized=assume_normalized, **kwargs,
    #     )
    #     self.aspiration_init(
    #         max_aspiration=max_aspiration, aspiration_type=aspiration_type
    #     )
    #     if not dynamic_ufun:
    #         warnings.warn(
    #             "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
    #         )
    #     self.randomize_offer = randomize_offer
    #     self._max_aspiration = self.max_aspiration
    #     self.best_outcome, self.worst_outcome = None, None
    #     self.presort = presort
    #     self.presorted = False
    #     self.add_capabilities(
    #         {
    #             "respond": True,
    #             "propose": can_propose,
    #             "propose-with-value": False,
    #             "max-proposals": None,  # indicates infinity
    #         }
    #     )
    #     self.__last_offer_util, self.__last_offer = float("inf"), None
    #     self.n_outcomes_to_force_presort = 10000
    #     self.n_trials = 1
    #
    # def on_ufun_changed(self):
    #     super().on_ufun_changed()
    #     presort = self.presort
    #     if (
    #         not presort
    #         and all(i.is_countable() for i in self._ami.issues)
    #         and Issue.num_outcomes(self._ami.issues) >= self.n_outcomes_to_force_presort
    #     ):
    #         presort = True
    #     if presort:
    #         outcomes = self._ami.discrete_outcomes()
    #         uvals = self.utility_function.eval_all(outcomes)
    #         uvals_outcomes = [
    #             (u, o)
    #             for u, o in zip(uvals, outcomes)
    #             if u >= self.utility_function.reserved_value
    #         ]
    #         self.ordered_outcomes = sorted(
    #             uvals_outcomes,
    #             key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
    #             reverse=True,
    #         )
    #         if self.assume_normalized:
    #             self.ufun_min, self.ufun_max = 0.0, 1.0
    #         elif len(self.ordered_outcomes) < 1:
    #             self.ufun_max = self.ufun_min = self.utility_function.reserved_value
    #         else:
    #             if self.ufun_max is None:
    #                 self.ufun_max = self.ordered_outcomes[0][0]
    #
    #             if self.ufun_min is None:
    #                 # we set the minimum utility to the minimum finite value above both reserved_value
    #                 for j in range(len(self.ordered_outcomes) - 1, -1, -1):
    #                     self.ufun_min = self.ordered_outcomes[j][0]
    #                     if self.ufun_min is not None and self.ufun_min > float("-inf"):
    #                         break
    #                 if (
    #                     self.ufun_min is not None
    #                     and self.ufun_min < self.reserved_value
    #                 ):
    #                     self.ufun_min = self.reserved_value
    #     else:
    #         if (
    #             self.ufun_min is None
    #             or self.ufun_max is None
    #             or self.best_outcome is None
    #             or self.worst_outcome is None
    #         ):
    #             mn, mx, self.worst_outcome, self.best_outcome = utility_range(
    #                 self.ufun, return_outcomes=True, issues=self._ami.issues
    #             )
    #             if self.ufun_min is None:
    #                 self.ufun_min = mn
    #             if self.ufun_max is None:
    #                 self.ufun_max = mx
    #
    #     if self.ufun_min < self.reserved_value:
    #         self.ufun_min = self.reserved_value
    #     if self.ufun_max < self.ufun_min:
    #         self.ufun_max = self.ufun_min
    #
    #     self.presorted = presort
    #     self.n_trials = 10
    #
    # def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
    #     if self.ufun_max is None or self.ufun_min is None:
    #         self.on_ufun_changed()
    #     if self._utility_function is None:
    #         return ResponseType.REJECT_OFFER
    #     u = self._utility_function(offer)
    #     if u is None or u < self.reserved_value:
    #         return ResponseType.REJECT_OFFER
    #     asp = (
    #         self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
    #         + self.ufun_min
    #     )
    #     if u >= asp and u > self.reserved_value:
    #         return ResponseType.ACCEPT_OFFER
    #     if asp < self.reserved_value:
    #         return ResponseType.END_NEGOTIATION
    #     return ResponseType.REJECT_OFFER
    #
    # def propose(self, state: MechanismState) -> Optional["Outcome"]:
    #     if self.ufun_max is None or self.ufun_min is None:
    #         self.on_ufun_changed()
    #     if self.ufun_max < self.reserved_value:
    #         return None
    #     asp = (
    #         self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
    #         + self.ufun_min
    #     )
    #     if asp < self.reserved_value:
    #         return None
    #     if self.presorted:
    #         if len(self.ordered_outcomes) < 1:
    #             return None
    #         for i, (u, o) in enumerate(self.ordered_outcomes):
    #             if u is None:
    #                 continue
    #             if u < asp:
    #                 if u < self.reserved_value:
    #                     return None
    #                 if i == 0:
    #                     return self.ordered_outcomes[i][1]
    #                 if self.randomize_offer:
    #                     return random.sample(self.ordered_outcomes[:i], 1)[0][1]
    #                 return self.ordered_outcomes[i - 1][1]
    #         if self.randomize_offer:
    #             return random.sample(self.ordered_outcomes, 1)[0][1]
    #         return self.ordered_outcomes[-1][1]
    #     else:
    #         if asp >= 0.99999999999 and self.best_outcome is not None:
    #             return self.best_outcome
    #         if self.randomize_offer:
    #             return outcome_with_utility(
    #                 ufun=self._utility_function,
    #                 rng=(asp, float("inf")),
    #                 issues=self._ami.issues,
    #             )
    #         tol = self.tolerance
    #         for _ in range(self.n_trials):
    #             rng = self.ufun_max - self.ufun_min
    #             mx = min(asp + tol * rng, self.__last_offer_util)
    #             outcome = outcome_with_utility(
    #                 ufun=self._utility_function, rng=(asp, mx), issues=self._ami.issues,
    #             )
    #             if outcome is not None:
    #                 break
    #             tol = math.sqrt(tol)
    #         else:
    #             outcome = (
    #                 self.best_outcome
    #                 if self.__last_offer is None
    #                 else self.__last_offer
    #             )
    #         self.__last_offer_util = self.utility_function(outcome)
    #         self.__last_offer = outcome
    #         return outcome


########################################## END

    def __str__(self):
        return (
            f"{'selling' if self.is_seller else 'buying'} p{self.product} [{self.step}] "
            f"secured {self.secured} of {self.target} for {self.parent_name} "
            f"({len([_ for _ in self.completed.values() if _])} completed of {len(self.completed)} negotiators)"
        )

    def create_negotiator(
        self,
        negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> PassThroughNegotiator:
        neg = super().create_negotiator(negotiator_type, name, cntxt, **kwargs)
        self.completed[neg.id] = False
        return neg

    def time_range(self, step, is_seller):
        if is_seller:
            return (
                max(step, self.awi.current_step + 1),
                min(step + self.horizon, self.awi.n_steps - 1),
            )
        return self.awi.current_step + 1, step - 1

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        super().on_negotiation_end(negotiator_id, state)
        agreement = state.agreement
        # mark this negotiation as completed
        self.completed[negotiator_id] = True
        # if there is an agreement increase the secured amount and check if we are done.
        if agreement is not None:
            self.secured += agreement[QUANTITY]
            if self.secured >= self.target:
                self.awi.loginfo(f"Ending all negotiations on controller {str(self)}")
                # If we are done, end all other negotiations
                for k in self.negotiators.keys():
                    if self.completed[k]:
                        continue
                    self.notify(
                        self.negotiators[k][0], Notification("end_negotiation", None)
                    )
        self.kill_negotiator(negotiator_id, force=True)
        if all(self.completed.values()):
            # If we secured everything, just return control to the agent
            if self.secured >= self.target:
                self.awi.loginfo(f"Secured Everything: {str(self)}")
                self.negotiations_concluded_callback(self.step, self.is_seller)
                return
            # If we did not secure everything we need yet and time allows it, create new negotiations
            tmin, tmax = self.time_range(self.step, self.is_seller)

            if self.awi.current_step < tmax + 1 and tmin <= tmax:
                # get a good partner: one that was not retired too much

                # # todo: MADE A CHANGE
                # possible_partners = [p for p in self.partners if self.retries[p] <= self.max_retries]
                # for partner in possible_partners:
                #     self.retries[partner] += 1
                #     neg = self.create_negotiator()
                #     self.completed[neg.id] = False
                #     self.awi.loginfo(
                #         f"{str(self)} negotiating with {partner} on u={self.urange}"
                #         f", q=(1,{self.target - self.secured}), u=({tmin}, {tmax})"
                #     )
                #     self.awi.request_negotiation(
                #         not self.is_seller,
                #         product=self.product,
                #         quantity=(1, self.target - self.secured),
                #         unit_price=self.urange,
                #         time=(tmin, tmax),
                #         partner=partner,
                #         negotiator=neg,
                #         extra=dict(controller_index=self.step, is_seller=self.is_seller),
                #     )

                random.shuffle(self.partners)
                for other in self.partners:
                    if self.retries[other] <= self.max_retries:
                        partner = other
                        break
                else:
                    return
                self.retries[partner] += 1
                neg = self.create_negotiator()
                self.completed[neg.id] = False
                self.awi.loginfo(
                    f"{str(self)} negotiating with {partner} on u={self.urange}"
                    f", q=(1,{self.target-self.secured}), u=({tmin}, {tmax})"
                )
                self.awi.request_negotiation(
                    not self.is_seller,
                    product=self.product,
                    quantity=(1, self.target - self.secured),
                    unit_price=self.urange,
                    time=(tmin, tmax),
                    partner=partner,
                    negotiator=neg,
                    extra=dict(controller_index=self.step, is_seller=self.is_seller),
                )


""" implementation of SyncController for every step """

#
# class StepSyncController(SAOSyncController):
#     """
#     Will try to get the best deal which is defined as being nearest to the agent needs and with lowest price
#     """
#
#     def __init__(
#         self,
#         *args,
#         is_seller: bool,
#         price_weight=0.7,
#         utility_threshold=0.9,
#         time_threshold=0.9,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self._is_seller = is_seller
#         self._time_threshold = time_threshold
#         self._price_weight = price_weight
#         self._utility_threshold = utility_threshold
#         self._best_utils: Dict[str, float] = {}
#         # find out my needs and the amount secured lists
#
#     def utility(self, offer: Tuple[int, int, int], max_price: int) -> float:
#         """A simple utility function
#
#         Remarks:
#              - If the time is invalid or there is no need to get any more agreements
#                at the given time, return -1000
#              - Otherwise use the price-weight to calculate a linear combination of
#                the price and the how much of the needs is satisfied by this contract
#
#         """
#         # print("UTILITY")
#         return self.ufun(offer)
#         # if self._is_seller:
#         #     _needed, _secured = (
#         #         self.target,
#         #         self.secured,
#         #     )
#         # else:
#         #     _needed, _secured = (
#         #         self.target,
#         #         self.secured,
#         #     )
#         # if offer is None:
#         #     return -1000
#         # t = offer[TIME]
#         # if t < self.awi.current_step or t > self.awi.n_steps - 1:
#         #     return -1000.0
#         # q = _needed - (offer[QUANTITY] + _secured)
#         # if q < 0:
#         #     return -1000.0
#         # if self._is_seller:
#         #     price = offer[UNIT_PRICE]
#         # else:
#         #     price = max_price - offer[UNIT_PRICE]
#         # return self._price_weight * price + (1 - self._price_weight) * q
#
#     def is_valid(self, negotiator_id: str, offer: "Outcome") -> bool:
#         issues = self.negotiators[negotiator_id][0].ami.issues
#         return outcome_is_valid(offer, issues)
#
#     def counter_all(
#         self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
#     ) -> Dict[str, SAOResponse]:
#         """Calculate a response to all offers from all negotiators (negotiator ID is the key).
#
#         Args:
#             offers: Maps negotiator IDs to offers
#             states: Maps negotiator IDs to offers AT the time the offers were made.
#
#         Remarks:
#             - The response type CANNOT be WAIT.
#
#         """
#
#         # find the best offer
#         negotiator_ids = list(offers.keys())
#         utils = np.array(
#             [self.utility(o, self.negotiators[nid][0].ami.issues[UNIT_PRICE].max_value)
#              for nid, o in offers.items()]
#         )
#
#         best_index = int(np.argmax(utils))
#         best_utility = utils[best_index]
#         best_partner = negotiator_ids[best_index]
#         best_offer = offers[best_partner]
#
#         # find my best proposal for each negotiation
#         best_proposals = self.first_proposals()
#
#         # if the best offer is still so bad just reject everything
#         if best_utility < 0:
#             return {
#                 k: SAOResponse(ResponseType.REJECT_OFFER, best_proposals[k])
#                 for k in offers.keys()
#             }
#
#         relative_time = min(_.relative_time for _ in states.values())
#
#         # if this is good enough or the negotiation is about to end accept the best offer
#         if (
#             best_utility >= self._utility_threshold * self._best_utils[best_partner]
#             or relative_time > self._time_threshold
#         ):
#             responses = {
#                 k: SAOResponse(
#                     ResponseType.REJECT_OFFER,
#                     best_offer if self.is_valid(k, best_offer) else best_proposals[k],
#                 )
#                 for k in offers.keys()
#             }
#             responses[best_partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
#             return responses
#
#         # send the best offer to everyone else and try to improve it
#         responses = {
#             k: SAOResponse(
#                 ResponseType.REJECT_OFFER,
#                 best_offer if self.is_valid(k, best_offer) else best_proposals[k],
#             )
#             for k in offers.keys()
#         }
#         responses[best_partner] = SAOResponse(
#             ResponseType.REJECT_OFFER, best_proposals[best_partner]
#         )
#         return responses
#
#     # def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
#     #     """Update the secured quantities whenever a negotiation ends"""
#     #     if state.agreement is None:
#     #         return
#     #
#     #     q, t = state.agreement[QUANTITY], state.agreement[TIME]
#     #     if self._is_seller:
#     #         self.__parent.outputs_secured[t] += q
#     #     else:
#     #         self.__parent.inputs_secured[t] += q
#
#     def best_proposal(self, nid: str) -> Tuple[Optional[Outcome], float]:
#         """
#         Finds the best proposal for the given negotiation
#
#         Args:
#             nid: Negotiator ID
#
#         Returns:
#             The outcome with highest utility and the corresponding utility
#         """
#         negotiator = self.negotiators[nid][0]
#         if negotiator.ami is None:
#             return None, -1000
#         utils = np.array(
#             [
#                 self.utility(_, negotiator.ami.issues[UNIT_PRICE].max_value)
#                 for _ in negotiator.ami.outcomes
#             ]
#         )
#         best_indx = np.argmax(utils)
#         self._best_utils[nid] = utils[best_indx]
#         if utils[best_indx] < 0:
#             return None, utils[best_indx]
#         return negotiator.ami.outcomes[best_indx], utils[best_indx]
#
#     def first_proposals(self) -> Dict[str, "Outcome"]:
#         """Gets a set of proposals to use for initializing the negotiation."""
#         return {nid: self.best_proposal(nid)[0] for nid in self.negotiators.keys()}
#
#
# class DanasController(StepSyncController, AspirationMixin, Notifier):
#     """A controller for managing a set of negotiations about selling or buying (but not both)  starting/ending at some
#     specific time-step.
#
#     Args:
#
#         target_quantity: The quantity to be secured
#         is_seller:  Is this a seller or a buyer
#         parent_name: Name of the parent
#         horizon: How many steps in the future to allow negotiations for selling to go for.
#         step:  The simulation step that this controller is responsible about
#         urange: The range of unit prices used for negotiation
#         product: The product that this controller negotiates about
#         partners: A list of partners to negotiate with
#         negotiator_type: The type of the single negotiator used for all negotiations.
#         negotiator_params: The parameters of the negotiator used for all negotiations
#         max_retries: How many times can the controller try negotiating with each partner.
#         negotiations_concluded_callback: A method to be called with the step of this controller and whether is it a
#                                          seller when all negotiations are concluded
#         *args: Position arguments passed to the base Controller constructor
#         **kwargs: Keyword arguments passed to the base Controller constructor
#
#
#     Remarks:
#
#         - It uses whatever negotiator type on all of its negotiations and it assumes that the ufun will never change
#         - Once it accumulates the required quantity, it ends all remaining negotiations
#         - It assumes that all ufuns are identical so there is no need to keep a separate negotiator for each one and it
#           instantiates a single negotiator that dynamically changes the AMI but always uses the same ufun.
#
#     """
#
#     def __init__(
#         self,
#         *args,
#         target_quantity: int,
#         is_seller: bool,
#         step: int,
#         urange: Tuple[int, int],
#         product: int,
#         partners: List[str],
#         negotiator_type: SAONegotiator,
#         horizon: int,
#         awi: AgentWorldInterface,
#         parent_name: str,
#         negotiations_concluded_callback: Callable[[int, bool], None],
#         negotiator_params: Dict[str, Any] = None,
#         max_retries: int = 2,
#         **kwargs,
#     ):
#         super().__init__(*args, is_seller=is_seller, **kwargs)
#         self.parent_name = parent_name
#         self.awi = awi
#         self.horizon = horizon
#         self.negotiations_concluded_callback = negotiations_concluded_callback
#         self.is_seller = is_seller
#         self.target = target_quantity
#         self.urange = urange
#         self.partners = partners
#         self.product = product
#         # negotiator_params = (
#         #     negotiator_params if negotiator_params is not None else dict()
#         # )
#         self.secured = 0
#         if is_seller:
#             self.ufun = DanasUtilityFunction((1, 1, 10))
#         else:
#             self.ufun = DanasUtilityFunction((1, -1, -10))
#         # negotiator_params["ufun"] = self.ufun
#         # self.__negotiator = instantiate(negotiator_type, **negotiator_params)
#         self.completed = defaultdict(bool)
#         self.step = step
#         self.retries: Dict[str, int] = defaultdict(int)
#         self.max_retries = max_retries
#
#     # def join(
#     #     self,
#     #     negotiator_id: str,
#     #     ami: AgentMechanismInterface,
#     #     state: MechanismState,
#     #     *,
#     #     ufun: Optional["UtilityFunction"] = None,
#     #     role: str = "agent",
#     # ) -> bool:
#     #     raise Exception
#         # joined = super().join(negotiator_id, ami, state, ufun=ufun, role=role)
#         # if joined:
#         #     self.completed[negotiator_id] = False
#         # return joined
#
#     # def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
#     #     self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
#     #     return self.__negotiator.propose(state)
#
#     # def respond(
#     #     self, negotiator_id: str, state: MechanismState, offer: "Outcome"
#     # ) -> ResponseType:
#     #     if self.secured >= self.target:
#     #         return ResponseType.END_NEGOTIATION
#     #     self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
#     #     return self.__negotiator.respond(offer=offer, state=state)
#
#     def __str__(self):
#         return (
#             f"{'selling' if self.is_seller else 'buying'} p{self.product} [{self.step}] "
#             f"secured {self.secured} of {self.target} for {self.parent_name} "
#             f"({len([_ for _ in self.completed.values() if _])} completed of {len(self.completed)} negotiators)"
#         )
#
#     def create_negotiator(
#         self,
#         *args,
#         # negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
#         # name: str = None,
#         # cntxt: Any = None,
#         **kwargs,
#     ) -> PassThroughNegotiator:
#         neg = super().create_negotiator(*args, **kwargs)  # negotiator_type, name, cntxt, **kwargs)
#         self.completed[neg.id] = False
#         return neg
#
#     def time_range(self, step, is_seller):
#         if is_seller:
#             return (
#                 max(step, self.awi.current_step + 1),
#                 min(step + self.horizon, self.awi.n_steps - 1),
#             )
#         return self.awi.current_step + 1, step - 1
#
#     def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
#         super().on_negotiation_end(negotiator_id, state)
#         agreement = state.agreement
#         # mark this negotiation as completed
#         self.completed[negotiator_id] = True
#         # if there is an agreement increase the secured amount and check if we are done.
#         if agreement is not None:
#             self.secured += agreement[QUANTITY]
#             if self.secured >= self.target:
#                 self.awi.loginfo(f"Ending all negotiations on controller {str(self)}")
#                 # If we are done, end all other negotiations
#                 for k in self.negotiators.keys():
#                     if self.completed[k]:
#                         continue
#                     self.notify(
#                         self.negotiators[k][0], Notification("end_negotiation", None)
#                     )
#         self.kill_negotiator(negotiator_id, force=True)
#         if all(self.completed.values()):
#             # If we secured everything, just return control to the agent
#             if self.secured >= self.target:
#                 self.awi.loginfo(f"Secured Everything: {str(self)}")
#                 self.negotiations_concluded_callback(self.step, self.is_seller)
#                 return
#             # If we did not secure everything we need yet and time allows it, create new negotiations
#             tmin, tmax = self.time_range(self.step, self.is_seller)
#
#             if self.awi.current_step < tmax + 1 and tmin <= tmax:
#                 # get a good partner: one that was not retired too much
#                 random.shuffle(self.partners)
#                 for other in self.partners:
#                     if self.retries[other] <= self.max_retries:
#                         partner = other
#                         break
#                 else:
#                     return
#                 self.retries[partner] += 1
#                 neg = self.create_negotiator()
#                 self.completed[neg.id] = False
#                 self.awi.loginfo(
#                     f"{str(self)} negotiating with {partner} on u={self.urange}"
#                     f", q=(1,{self.target-self.secured}), u=({tmin}, {tmax})"
#                 )
#                 self.awi.request_negotiation(
#                     not self.is_seller,
#                     product=self.product,
#                     quantity=(1, self.target - self.secured),
#                     unit_price=self.urange,
#                     time=(tmin, tmax),
#                     partner=partner,
#                     negotiator=neg,
#                     extra=dict(controller_index=self.step, is_seller=self.is_seller),
#                 )


"""
improvements:
1) StepController, on_neg_end - send multiple nrg requests (bad)
2) use a SyncController with StepController's on_neg_end (bad)
"""