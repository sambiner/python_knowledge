"""
Skeleton for answering queries related to the Ad Agent.

@author: Sam Biner, Brandon Bazile, and Joanna Estrada
"""

from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from ad_agent import AdEngine
from constants import *
import numpy as np
import pandas as pd


class AdAgentQueries:
    """
    See Problem 7 in the Spec for requested answer formats below
    """

    def __init__(self, ad_agent: "AdEngine") -> None:
        self._ad_agent = ad_agent

    def answer_7_1(self) -> float:
        return self._ad_agent.vpi("G", {})

    def answer_7_2(self) -> float:
        return self._ad_agent.vpi("P", {"G": 1})

    def answer_7_3(self) -> dict[str, int]:
        return self._ad_agent.most_likely_consumer({"A": 0, "H": 1})


if __name__ == "__main__":
    """
    Use this main method to run the requested queries for your report
    """
    # Initialize Ad Agent and Query Helper Class
    ad_agent = AdEngine(ADBOT_DATA, ADBOT_STRUC, ADBOT_DEC, ADBOT_UTIL)
    querier = AdAgentQueries(ad_agent)

    # 7.1 Answer: 20.7794
    # This means that, in the context of price, that the juice would be worth the squeeze
    # if the information costs less than $20.78
    print("Answer to 7.1: " + str(querier.answer_7_1()))

    # 7.2 Answer: 0.0
    # This means it gives no value to know a customer's Political Affiliation when we
    # know they are in support of Gun Control
    print("Answer to 7.2: " + str(querier.answer_7_2()))

    # 7.3 Answer: {'I': 1, 'T': 1, 'P': 1, 'F': 0, 'S': 0, 'G': 1, 'A': 0, 'H': 1}
    # The customer posesses insurance, perceives a threat of crime, is a liberal, does not perceive a threat of
    # foreign invasion, does not buy anything, and supports gun control in addition to being a millenial and home owner
    print("Answer to 7.3: " + str(querier.answer_7_3()))
