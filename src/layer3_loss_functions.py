import numpy as np
import sys
sys.path.append('.')

def compute_voltage_mse(simulated_voltage, target_voltage):
    """
    Compute Mean Squared Error between voltage traces.

    This is the most basic loss - compare voltage point by point.

    MSE = (1/N) × Σ(v_sim - v_target)²
    reason to get the square - otherwise it will cancel each other out.

    Args:
        simulated_voltage (np.ndarray): Simulated voltage trace
        target_voltage (np.ndarray): Target voltage trace
    """
    assert len(simulated_voltage)==len(target_voltage), \
        "Voltage traces must have some length!"
    
    squared_diff = (simulated_voltage- target_voltage)**2

    mse = np.mean(squared_diff)
    return mse

def compute_subthreshold_mse(simulated_voltage, target_voltage, params):
    """
    Compute MSE only on differniable parts - excluding spikes 
     this is important because 
     - spikes are discontinous 

    Args:
        simulated_voltage (np.ndarray): simulated voltage trace
        target_voltage (np.ndarray): target voltage trace
        params (dict): neuron parameters
    """
    v_reset = params['v_reset']

    #Create a mask - true where voltage is not at reset
    sim_mask = simulated_voltage != v_reset
    target_mask = target_voltage != v_reset

    #only include points that are sub-threshold in both traces
    combined_mask = sim_mask& target_mask

    #extract sub threshold voltages
    sim_subthreshold = simulated_voltage[combined_mask]
    target_subthreshold = target_voltage[combined_mask]

    #compute MSE only in these points
    if len(sim_subthreshold) == 0:
        return 0.0
    
    squared_diff = (sim_subthreshold - target_subthreshold)**2
    mse = np.mean(squared_diff)

    return mse