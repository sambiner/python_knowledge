"""
Constants to be used across various Ad Agent and Decision Network
tests, including network structure configurations and test file locations.

[!] Feel free to edit this file at will
[!] Note: any data frame in here should not be modified!
"""

import pandas as pd

# Lecture 5-2 Constants
# -----------------------------------------------------------------------------------------
LECTURE_5_2_DATA = pd.read_csv(
    "C:/CMSI_3300/lmu-cmsi3300-fall2023-homework2-sam_brandon_joanna/dat/lecture5-2-data.csv"
)
LECTURE_5_2_STRUC = [("M", "C"), ("D", "C")]
LECTURE_5_2_DEC = ["D"]
LECTURE_5_2_UTIL = {"C": {0: 3, 1: 1}}

# AdBot Constants
# -----------------------------------------------------------------------------------------
ADBOT_DATA = pd.read_csv(
    "C:/CMSI_3300/lmu-cmsi3300-fall2023-homework2-sam_brandon_joanna/dat/adbot-data.csv"
)
ADBOT_STRUC = [
    ("H", "I"),
    ("A", "H"),
    ("A", "F"),
    ("Ad2", "F"),
    ("A", "T"),
    ("P", "T"),
    ("P", "G"),
    ("F", "S"),
    ("Ad1", "G"),
    ("Ad1", "S"),
    ("G", "S"),
]
ADBOT_DEC = ["Ad1", "Ad2"]
ADBOT_UTIL = {"S": {0: 0, 1: 1776, 2: 500}}
