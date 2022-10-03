# aux.py

"""This module defines auxiliary functions used in the metamaterials package."""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

def find_nearest(array, value):
    """Find nearest value in array and returns its index."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
