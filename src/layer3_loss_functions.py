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
