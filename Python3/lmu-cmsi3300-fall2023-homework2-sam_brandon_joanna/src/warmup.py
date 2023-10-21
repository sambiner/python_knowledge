"""
warmup.py

Skeleton for answering warmup questions related to the
AdAgent assignment. By the end of this section, you should
be familiar with:
- Importing, selecting, and manipulating data using Pandas
- Creating and Querying a Bayesian Network
- Using Samples from a Bayesian Network for Approximate Inference

@author: Sam Biner, Brandon Bazile, and Joanna Estrada
"""
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd

if __name__ == "__main__":
    """
    PROBLEM 2
    Using the pgmpy query example, determine the answers to the
    queries specified in the instructions.

    (just print out the CPT values with their labels and save to report)
    """
    # Load the data into a pandas data frame
    csv_data = pd.read_csv(
        "C:/CMSI_3300/lmu-cmsi3300-fall2023-homework2-sam_brandon_joanna/dat/warmup-data.csv"
    )

    # Set the edges of the network: tuples of the format (parent, child)
    edges = [("W", "X"), ("W", "Y"), ("X", "Z"), ("Y", "Z")]

    # Build the network structure from the edges
    model = BayesianNetwork(edges)

    # "Fit" the model = learn the CPTs from the data and structure
    model.fit(csv_data)

    # Create the inference engine using the Variable Elimination algorithm
    # (a more efficient enumeration inference)
    inference = VariableElimination(model)

    # Here's an example query: P(X | W=0)
    query = inference.query(["W"], evidence={})

    # Note the CPT is given when the query is printed
    print(query)

    # And to programmatically access those, you can access the values attribute
    # of the CPT and then grab any by index, e.g.,
    print("P(X | W=0) = " + str(query.values[0]))


# Problem 2 Answers:
# P(W) = 0.3013
# P(X | W = 0) = 0.19914
# P(Z | W = 0, X = 0) = 0.2615
