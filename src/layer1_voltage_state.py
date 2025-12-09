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
        #stops program if bad input
        raise ValueError(f"Unknown condition: {condition}. Use 'rest', 'depolarized', 'hyperpolarized', or 'threshold'")
    
def plot_voltage_state(time, voltage, params, title="Voltage state"):
    """
    Plot voltage over time with reference lines

    Args:
        time (np.ndarray): Time grid
        voltage (np.ndarray): Volatge values
        params (dict): Neuron parameters (for reference lines)
        title (str): Plot title
    """
    plt.figure(figsize=(12,6))

    # Add reference lines for key voltages
    plt.axhline(y=params['v_rest'], color='blue', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='V_rest')
    plt.axhline(y=params['v_threshold'], color='red', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='V_threshold')
    plt.axhline(y=params['v_reset'], color='green', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='V_reset')
    
    # Labels and formatting
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Voltage (mV)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()