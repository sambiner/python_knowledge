"""
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.

Solution Amended from Legendary N2A Team
> Warning: not a great amendment: was just playing with some
  settings to get the new pgmpy library working
"""
import math
import itertools
import unittest
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork


class AdEngine:
    def __init__(
        self,
        data: "pd.DataFrame",
        structure: list[tuple[str, str]],
        dec_vars: list[str],
        util_map: dict[str, dict[int, int]],
    ):
        """
        Responsible for initializing the Decision Network of the
        AdEngine by taking in the dataset, structure of the network,
        any decision variables, and a map of utilities
        Parameters:
            data (pd.DataFrame):
                Pandas data frame containing all data on which the decision
                network's chance-node parameters are to be learned
            structure (list[tuple[str, str]]):
                The Bayesian Network's structure, a list of tuples denoting
                the edge directions where each tuple is (parent, child)
            dec_vars (list[str]):
                list of string names of variables to be
                considered decision variables for the agent. Example:
                ["Ad1", "Ad2"]
            util_map (dict[str, dict[int, int]]):
                Discrete, tabular, utility map whose keys
                are variables in network that are parents of a utility node, and
                values are dictionaries mapping that variable's values to a utility
                score, for example:
                  {"X": {0: 20, 1: -10}}
                represents a utility node with single parent X whose value of 0
                has a utility score of 20, and value 1 has a utility score of -10
        """
        self.model = BayesianNetwork(structure)
        self.model.fit(data, estimator=MaximumLikelihoodEstimator)

        self.dec_vars = dec_vars
        self.util_map = util_map

        self.chance_vars = [
            var
            for var in data.columns
            if var not in dec_vars and var not in util_map.keys()
        ]
        self.unique_values = {var: data[var].unique().tolist() for var in data.columns}

    def meu(self, evidence: dict[str, int]) -> tuple[dict[str, int], float]:
        """
        Computes the Maximum Expected Utility (MEU) defined as the choice of
        decision variable values that maximize expected utility of any evaluated
        chance nodes given in the agent's utility map.
        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables to their observed values,
                of the format: {"Obs1": val1, "Obs2": val2, ...}
        Returns:
            tuple[dict[str, int], float]:
                A 2-tuple of the format (a*, MEU) where:
                [0] is a dictionary mapping decision variables to their MEU states
                [1] is the MEU value (a float) of that decision combo
        """
        best_action = {}
        max_utility = float("-inf")
        decision_space = [self.unique_values[var] for var in self.dec_vars]

        for action_combo in itertools.product(*decision_space):
            action = dict(zip(self.dec_vars, action_combo))
            expected_utility = self.calculate_expected_utility(action, evidence)

            if expected_utility > max_utility:
                best_action = action
                max_utility = expected_utility

        return best_action, max_utility

    def calculate_expected_utility(
        self, action: dict[str, int], evidence: dict[str, int]
    ) -> float:
        combined_evidence = {**action, **evidence}
        inference = VariableElimination(self.model)

        expected_utility = 0
        for util_var in self.util_map.keys():
            query_result = inference.query(
                variables=[util_var], evidence=combined_evidence
            )
            probabilities = query_result.values
            states = query_result.state_names[util_var]
            for i, state in enumerate(states):
                util = self.util_map[util_var].get(state, 0)
                expected_utility += util * probabilities[i]

        return expected_utility

    def vpi(self, potential_evidence: str, observed_evidence: dict[str, int]) -> float:
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.
        Parameters:
            potential_evidence (str):
                string representing the variable name of the variable
                under consideration for potentially being obtained
            observed_evidence (tuple[dict[str, int], float]):
                dict mapping network variables
                to their observed values, of the format:
                {"Obs1": val1, "Obs2": val2, ...}
        Returns:
            float:
                float value indicating the VPI(potential | observed)
        """
        _, meu_without_evidence = self.meu(observed_evidence)
        sum_meu_with_evidence = 0

        inference = VariableElimination(self.model)
        potential_evidence_distribution = inference.query(
            [potential_evidence], evidence=observed_evidence
        )
        probabilities = potential_evidence_distribution.values
        states = potential_evidence_distribution.state_names[potential_evidence]
        for i, state in enumerate(states):
            prob = probabilities[i]
            new_evidence = {**observed_evidence, potential_evidence: state}
            _, meu_with_new_evidence = self.meu(new_evidence)
            sum_meu_with_evidence += prob * meu_with_new_evidence

        vpi_value = sum_meu_with_evidence - meu_without_evidence
        return max(vpi_value, 0.0)

    def most_likely_consumer(self, evidence: dict[str, int]) -> dict[str, int]:
        """
        Given some known traits about a particular consumer, makes the best guess
        of the values of any remaining hidden variables and returns the completed
        data point as a dictionary of variables mapped to their most likely values.
        (Observed evidence will always have the same values in the output).
        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables
                to their observed values, of the format:
                {"Obs1": val1, "Obs2": val2, ...}
        Returns:
            dict[str, int]:
                The most likely values of all variables given what's already
                known about the consumer.
        """
        inference = VariableElimination(self.model)
        all_vars = set(self.model.nodes())
        observed_vars = set(evidence.keys())
        decision_vars = set(self.dec_vars)
        hidden_vars = list(all_vars - observed_vars - decision_vars)
        most_likely_values: dict[str, int] = inference.map_query(
            variables=hidden_vars, evidence=evidence
        )
        most_likely_values.update(evidence)
        return most_likely_values
