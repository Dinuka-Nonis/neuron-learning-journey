import numpy as np
import matplotlib.pyplot as plt

def initialize_voltage_state(time, v_initial):
    """
    Initialize voltage array for simulation.

    This creates a voltage array of the correct length and sets
    the initial voltage. Thes rest of the array is zerros as placeholders
    that will be filled during simulation.

    Args:
        time (np.ndarray): Time grid from laye1_input
        v_initial (float): Initial voltage value ine mV

    Returns:
        np.ndarray: voltage array with first element set to v_initial
    """

    voltage = np.zeros_like(time)
    voltage[0] = v_initial
    return voltage