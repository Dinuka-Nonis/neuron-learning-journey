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

def get_initial_conditions(params, condition='rest'):
    """
    Get common initial voltage conditions.

    Args:
        params (dict): Neuron parameters from layer1_parameters
        condition (str): Type of initial condition
            - 'rest': Start at resting potential (default)
            - 'depolarized': Start closer to threshold
            - 'hyperpolarized': Start below rest (post-spike like)
            - 'threshold': Start exactly at threshold

    Returns:
        float: Initial voltage in mV
    """

    if condition == 'rest':  # most common - neuron at rest
        return params['v_rest']
    elif condition == 'depolarized':  # partially excited, between rest and threshhold
        v_rest = params['v_rest']
        v_threshold = params['v_threshhold']
        return (v_rest + v_threshold)/2  # to get the midpoint
    elif condition == 'hyperpolarized':
        return params['v_reset']
    elif condition == 'threshold':
        return params['v_threshold']
    
    else:
        raise ValueError(f"Unknown condition: {condition}. Use 'rest', 'depolarized', 'hyperpolarized', or 'threshold'")