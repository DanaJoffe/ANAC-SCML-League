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
# from negmas.helpers import instantiate
# from negmas.common import AgentMechanismInterface
# from negmas.sao import (
#     SAOController,
#     SAONegotiator,
#     SAOSyncController,
# )

import functools
from negmas.events import Notifier
from scml import SCML2020Agent
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020.components.negotiation import ControllerInfo
from scml.scml2020.services import StepController
from run_tournament import run


# class MyAgent2(SCML2020Agent):
#     pass
"""
**Submitted to ANAC 2020 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to 
the authors and the ANAC 2020 SCML. 

This module implements a factory manager for the SCM 2020 league of ANAC 2019 
competition. This version will not use subcomponents. Please refer to the 
[game description](http://www.yasserm.com/scml/scml2020.pdf) for all the 
callbacks and subcomponents available.

Your git_scml_project can learn about the state of the world and itself by accessing
properties in the AWI it has. For example:

- The number of simulation steps (days): self.awi.n_steps  
- The current step (day): self.awi.current_steps
- The factory state: self.awi.state
- Availability for producton: self.awi.available_for_production


Your git_scml_project can act in the world by calling methods in the AWI it has.
For example:

- *self.awi.request_negotiation(...)*  # requests a negotiation with one partner
- *self.awi.request_negotiations(...)* # requests a set of negotiations


You can access the full list of these capabilities on the documentation.

- For properties/methods available only to SCM agents, check the list 
  [here](https://scml.readthedocs.io/en/latest/api/scml.scml2020.AWI.html)

"""

# required for running the test tournament
import time
# required for typing
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from negmas import (AgentMechanismInterface, Breach, Contract, Issue,
                    MechanismState, Negotiator, SAONegotiator, AspirationNegotiator, LinearUtilityFunction,
                    SAOController, AspirationMixin, AgentWorldInterface)
from negmas.helpers import humanize_time, instantiate
from scml.scml2020 import SCML2020Agent, PredictionBasedTradingStrategy, SupplyDrivenProductionStrategy, \
    StepNegotiationManager
# from scml.scml2020.components import (
#     SupplyDrivenProductionStrategy,
#     StepNegotiationManager,
#     IndependentNegotiationsManager,
# )



# required for development
from scml.scml2020.agents import (BuyCheapSellExpensiveAgent,
                                  DecentralizingAgent, DoNothingAgent)
from scml.scml2020.utils import anac2020_collusion, anac2020_std
from scml.scml2020.world import Failure, QUANTITY
from tabulate import tabulate





class DanasUtilityFunction(LinearUtilityFunction):
    pass


class DanasNegotiator(AspirationNegotiator):
    def __init__(
            self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # print("===> Dana's Negotiator!!")


# class DanasController(StepController):
#     def __init__(
#             self,
#             *args,
#             is_seller: bool,
#             # step: int,
#             # urange: Tuple[int, int],
#             # product: int,
#             # partners: List[str],
#             negotiator_type: SAONegotiator,
#             # horizon: int,
#             # awi: AgentWorldInterface,
#             # parent_name: str,
#             # negotiations_concluded_callback: Callable[[int, bool], None],
#             negotiator_params: Dict[str, Any] = None,
#             **kwargs,
#     ):
#         super().__init__(*args, is_seller=is_seller, negotiator_type=negotiator_type, **kwargs)
#         if is_seller:
#             self.ufun = DanasUtilityFunction((1, 1, 10))
#         else:
#             self.ufun = DanasUtilityFunction((1, -1, -10))
#         negotiator_params["ufun"] = self.ufun
#         self.__negotiator = instantiate(negotiator_type, **negotiator_params)
#
#     def create_negotiator(self, *args, **kwargs):
#         return super().create_negotiator(*args, **kwargs)


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
        negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )
        self.secured = 0
        # if is_seller:
        #     self.ufun = LinearUtilityFunction((1, 1, 10))
        # else:
        #     self.ufun = LinearUtilityFunction((1, -1, -10))
        if is_seller:
            self.ufun = DanasUtilityFunction((1, 1, 10))
        else:
            self.ufun = DanasUtilityFunction((1, -1, -10))
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
        return self.__negotiator.propose(state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> ResponseType:
        if self.secured >= self.target:
            return ResponseType.END_NEGOTIATION
        self.__negotiator._ami = self.negotiators[negotiator_id][0]._ami
        return self.__negotiator.respond(offer=offer, state=state)

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

