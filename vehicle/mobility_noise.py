### Module to generate noise on vehicles' location and testcases
import numpy as np


def add_random_noise_on_loc(x, y, mean_deviation=0, std_deviation=3.5):
    noise = np.random.normal(mean_deviation, std_deviation, 2)
    return max(0, x + noise[0]), max(0, y + noise[1]) # add sanity check >= 0 


