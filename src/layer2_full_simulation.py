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

def detect_spikes_from_voltage(voltage, time, params):
    """
    Detect spike by scanning through voltage array

    Args:
        voltage (np.ndarray): Voltage trace
        time (np.ndarray): v_reset(just resest)
        params (dict): Neuron parameters
    """
    v_threshold = params['v_threshold']
    v_reset = params['v_reset']

    spike_times = []

    for i in range (1, len(voltage)):
        if voltage[i] == v_reset and voltage[i-1]>= v_threshold:
            spike_times.append(time[i])

    return spike_times

def print_simulation_summary(voltage, spike_times, time_config, params):
    """
    print summary statistics from simulation

    Args:
        voltage (np.ndarray): voltage trace
        spike_times (list): times when spikes occurred
        time_config (dict): time configuration
        params (dict): neuron parameters
    """
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    # Voltage statistics
    print(f"\nVoltage Statistics:")
    print(f"  Min voltage: {np.min(voltage):.2f} mV")
    print(f"  Max voltage: {np.max(voltage):.2f} mV")
    print(f"  Mean voltage: {np.mean(voltage):.2f} mV")
    print(f"  Final voltage: {voltage[-1]:.2f} mV")
    
    # Spike statistics
    n_spikes = len(spike_times)
    print(f"\nSpike Statistics:")
    print(f"  Number of spikes: {n_spikes}")
    
    if n_spikes > 0:
        print(f"  First spike: {spike_times[0]:.2f} ms")
        print(f"  Last spike: {spike_times[-1]:.2f} ms")
        
        # Calculate firing rate
        t_total = time_config['t_total']
        firing_rate = (n_spikes / t_total) * 1000  # Convert to Hz
        print(f"  Mean firing rate: {firing_rate:.2f} Hz")
        
        # Calculate inter-spike intervals
        if n_spikes > 1:
            isis = np.diff(spike_times)  # Differences between consecutive spikes
            print(f"  Mean ISI: {np.mean(isis):.2f} ms")
            print(f"  ISI std: {np.std(isis):.2f} ms")
    else:
        print("  No spikes detected")
    
    print("="*60 + "\n")

def main():
    """
    Demonstrate full neuron simulation with different inputs.
    """
    from src.layer1_parameters import get_default_parameters
    from src.layer1_input import create_pulse_input, create_constant_inputs
    from src.layer1_time_representation import create_time_configuration
    
    # Setup
    params = get_default_parameters()
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    
    print("="*60)
    print("FULL NEURON SIMULATION EXAMPLES")
    print("="*60)
    
    # Example 1: Constant weak input (sub-threshold)
    print("\n📊 Example 1: Weak Constant Input (No Spikes)")
    print("-" * 60)
    current_weak = create_constant_inputs(time, amplitude=5.0)
    voltage_weak, spikes_weak = simulate_neuron_euler(
        params, time_config, current_weak, v_initial=params['v_rest']
    )
    print_simulation_summary(voltage_weak, spikes_weak, time_config, params)
    
    # Example 2: Constant strong input (supra-threshold)
    print("\n📊 Example 2: Strong Constant Input (Multiple Spikes)")
    print("-" * 60)
    current_strong = create_constant_inputs(time, amplitude=20.0)
    voltage_strong, spikes_strong = simulate_neuron_euler(
        params, time_config, current_strong, v_initial=params['v_rest']
    )
    print_simulation_summary(voltage_strong, spikes_strong, time_config, params)
    
    # Example 3: Pulse input
    print("\n📊 Example 3: Pulse Input (Transient Response)")
    print("-" * 60)
    current_pulse = create_pulse_input(time, start_time=20.0, end_time=60.0, amplitude=15.0)
    voltage_pulse, spikes_pulse = simulate_neuron_euler(
        params, time_config, current_pulse, v_initial=params['v_rest']
    )
    print_simulation_summary(voltage_pulse, spikes_pulse, time_config, params)
    
    # Verify spike detection matches
    print("\n🔍 Verification: Compare spike detection methods")
    print("-" * 60)
    spikes_detected = detect_spikes_from_voltage(voltage_strong, time, params)
    print(f"Spikes recorded during simulation: {len(spikes_strong)}")
    print(f"Spikes detected from voltage: {len(spikes_detected)}")
    print(f"Match: {len(spikes_strong) == len(spikes_detected)} ✓")


if __name__ == "__main__":
    main()