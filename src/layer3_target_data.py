import numpy as np
import sys
sys.path.append('.')
from src.layer1_parameters import get_default_parameters
from src.layer1_input import create_constant_inputs, create_pulse_input
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

def print_target_info(target_data):
    """
    Print summary information about target data.
    
    Args:
        target_data (dict): Target data from generate_target_data()
    """
    print("\n" + "="*60)
    print("TARGET DATA INFORMATION")
    print("="*60)
    
    # Time information
    time_config = target_data['time_config']
    print(f"\nTime Configuration:")
    print(f"  Duration: {time_config['t_total']:.1f} ms")
    print(f"  Time step: {time_config['dt']:.3f} ms")
    print(f"  Number of points: {time_config['n_steps']}")
    
    # True parameters (ground truth)
    params = target_data['params']
    print(f"\nTrue Parameters (Ground Truth):")
    print(f"  τ (tau): {params['tau']:.1f} ms")
    print(f"  V_rest: {params['v_rest']:.1f} mV")
    print(f"  V_threshold: {params['v_threshold']:.1f} mV")
    print(f"  V_reset: {params['v_reset']:.1f} mV")
    
    # Voltage statistics
    voltage = target_data['voltage']
    print(f"\nVoltage Statistics:")
    print(f"  Mean: {np.mean(voltage):.2f} mV")
    print(f"  Min: {np.min(voltage):.2f} mV")
    print(f"  Max: {np.max(voltage):.2f} mV")
    print(f"  Std: {np.std(voltage):.2f} mV")
    
    # Spike information
    spike_times = target_data['spike_times']
    n_spikes = len(spike_times)
    print(f"\nSpike Information:")
    print(f"  Number of spikes: {n_spikes}")
    
    if n_spikes > 0:
        print(f"  First spike: {spike_times[0]:.2f} ms")
        print(f"  Last spike: {spike_times[-1]:.2f} ms")
        
        firing_rate = (n_spikes / time_config['t_total']) * 1000
        print(f"  Firing rate: {firing_rate:.2f} Hz")
        
        if n_spikes > 1:
            isis = np.diff(spike_times)
            print(f"  Mean ISI: {np.mean(isis):.2f} ms")
            print(f"  ISI CV: {np.std(isis)/np.mean(isis):.3f}")
    
    # Input current statistics
    current = target_data['current']
    print(f"\nInput Current:")
    print(f"  Mean: {np.mean(current):.2f}")
    print(f"  Max: {np.max(current):.2f}")
    print(f"  Min: {np.min(current):.2f}")
    
    print("="*60 + "\n")

def main():
    """
    Demonstrate target data generation.
    """
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("TARGET DATA GENERATION EXAMPLES")
    print("="*60)
    
    # Setup
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    
    # Example 1: Target with constant input
    print("\n📊 Example 1: Constant Input Target")
    print("-" * 60)
    current_const = create_constant_inputs(time, amplitude=18.0)
    target_const = generate_target_data(current_const, time_config, noise_level=0.0)
    print_target_info(target_const)
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(target_const['time'], target_const['voltage'], 'b-', linewidth=2)
    for spike_time in target_const['spike_times']:
        plt.axvline(x=spike_time, color='red', linestyle=':', alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Example 1: Target Data (Constant Input)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Example 2: Target with pulse input
    print("\n📊 Example 2: Pulse Input Target")
    print("-" * 60)
    current_pulse = create_pulse_input(time, start_time=20.0, end_time=60.0, amplitude=25.0)
    target_pulse = generate_target_data(current_pulse, time_config, noise_level=0.0)
    print_target_info(target_pulse)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    ax1.plot(target_pulse['time'], target_pulse['voltage'], 'b-', linewidth=2)
    for spike_time in target_pulse['spike_times']:
        ax1.axvline(x=spike_time, color='red', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title('Example 2: Target Data (Pulse Input)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(target_pulse['time'], target_pulse['current'], 'orange', linewidth=2)
    ax2.fill_between(target_pulse['time'], 0, target_pulse['current'], alpha=0.3, color='orange')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Example 3: Target with noise
    print("\n📊 Example 3: Noisy Target (More Realistic)")
    print("-" * 60)
    target_noisy = generate_target_data(current_const, time_config, noise_level=1.0)
    print_target_info(target_noisy)
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    plt.plot(target_const['time'], target_const['voltage'], 'b-', 
             linewidth=2, alpha=0.5, label='Clean')
    plt.plot(target_noisy['time'], target_noisy['voltage'], 'r-', 
             linewidth=1, alpha=0.8, label='Noisy (σ=1.0 mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Example 3: Effect of Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Target data generation complete!")
    print("\nNote: These are the 'experimental recordings' we'll try to match.")
    print("The true parameters are known (for validation), but the learning")
    print("algorithm will NOT have access to them!")


if __name__ == "__main__":
    main()