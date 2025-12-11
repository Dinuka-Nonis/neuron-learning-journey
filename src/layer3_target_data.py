import numpy as np
import sys
sys.path.append('.')
from src.layer1_parameters import get_default_parameters
from src.layer1_input import create_constant_input, create_pulse_input
from src.layer1_time_representation import create_time_configuration
from src.layer2_full_simulation import simulate_neuron_euler

def generate_target_data(input_current, time_config, true_params=None, v_initial=None, noise_level=0.0):
    """
    Generate synthetic target data by simulating with "true" parameters.
    this creates the "experimental data" that we will try to match.

    Args:
        input_current (np.ndarray): input correction pattern
        time_config (dict): time configuaration
        true_params (dict, optional): the "true" neuron parameters. Defaults to None.
        v_initial (float, optional): Initial voltage. Defaults to None.
        noise_level (float, optional): standard deviation of Gaussian noise to add(mV). Defaults to 0.0.
    """

    #if no parameters provided, create slightly differnet oness

    if true_params is None:
        true_params = get_default_parameters()
        #modify slightly from defaults to make it interresting
        true_params['tau'] = 25.0
        true_params['v_threshold'] = -52.0
        true_params['v_reset'] = -75.0

    if v_initial is None:
        v_initial = true_params['v_rest']

    voltage, spike_times = simulate_neuron_euler(
        true_params, time_config, input_current, v_initial
    )

    if noise_level > 0 :
        voltage = add_noise_to_voltage(voltage, noise_level, true_params)

    target_data = {
        'voltage':voltage,
        'spike_times':spike_times,
        'current':input_current,
        'time': time_config['time'],
        'params': true_params.copy(),
        'time_config': time_config
    }

    return target_data

def add_noise_to_voltage(voltage, noise_level, params):
    """
    Add Gaussian noise to voltage trace (simulates experimental noise).
    
    Real neural recordings have noise from:
    - Electrode artifacts
    - Thermal noise
    - Biological variability
    
    Args:
        voltage (np.ndarray): Clean voltage trace
        noise_level (float): Standard deviation of noise (mV)
        params (dict): Neuron parameters (to ensure noise doesn't violate constraints)
    
    Returns:
        np.ndarray: Noisy voltage trace
    """
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, size=len(voltage))
    
    # Add to voltage
    noisy_voltage = voltage + noise
    
    # Optional: clip to reasonable range (prevent unrealistic values)
    v_reset = params['v_reset']
    v_threshold = params['v_threshold']
    
    # Don't let noise push voltage way outside normal range
    # Allow some margin beyond reset/threshold
    v_min = v_reset - 10  # 10 mV below reset
    v_max = v_threshold + 10  # 10 mV above threshold
    
    noisy_voltage = np.clip(noisy_voltage, v_min, v_max)
    
    return noisy_voltage