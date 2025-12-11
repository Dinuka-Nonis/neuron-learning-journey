import numpy as np
import sys
sys.path.append('.')
from src.layer2_single_step import simulate_single_step_with_spike

def simulate_neuron_euler(params, time_config, current, v_initial):
    """
    Simulate LIF neuron over time using Forwad euler method

    This is the main simulation loop. It:
    1. Initialize voltage array
    2. Steps through time
    3. Updates voltage at each steps 
    4. Detects and records spikes   

    Args:
        params (dict): Neuron parameters from layer1_parameters
        time_config (dict): Time configuration from layer1_time_representation
        current (np.ndarray): Input current array from laye1_input
        v_initial (float): Initial  voltage
    """

    #Extarct time information
    time = time_config['time']
    dt = time_config['dt']
    n_steps = time_config['n_steps']

    #Initialize voltage array
    voltage = np.zeros(n_steps)
    voltage[0] = v_initial

    #initialize spike recording
    spike_times = []

    # main simulation loop
    for i in range(n_steps -1):
        #Get current state
        V_current = voltage[i]
        I_current = current[i]

        #simulate one time step
        V_next, spike_occurred = simulate_single_step_with_spike(
            V_current, I_current, params, dt
        )

        #store next voltage
        voltage[i+1] = V_next

        #Record spike if occurred
        if spike_occurred:
            spike_times.append(time[i+1])

    return voltage, spike_times