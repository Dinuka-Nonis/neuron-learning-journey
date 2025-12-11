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
    if len(sim_subthreshold) == 0:  #if there are no such areas
        return 0.0
    
    squared_diff = (sim_subthreshold - target_subthreshold)**2
    mse = np.mean(squared_diff)

    return mse

def compute_spike_count_loss(simulated_spikes, target_spikes):
    """
    Compute the difference in number of spikes

    Args:
        simulated_spikes (list): simulated spike times
        target_spikes (list): target sike times
    """
    n_sim = len(simulated_spikes)
    n_target = len(target_spikes)

    loss = abs(n_sim- n_target)

    return float(loss)

def compute_spike_timing_loss(simulated_spikes, target_spikes, max_delay=10.0):
    """
    Compute spike timing missmatch
    using a smplified matching algorithm
     - for each target spike, find closest simulated spike
     - compute time difference
     - penalize large differences

    Args:
        simulated_spikes (list): simulated spike times
        target_spikes (list): target spike times
        max_delay (float, optional): maximum time difference. Defaults to 10.0.
    """ 

    n_sim = len(simulated_spikes)
    n_target = len(target_spikes)

    if n_sim == 0 and n_target == 0:
        return 0.0
    
    if n_sim == 0 or n_target ==0:
        return max_delay**2*max(n_sim, n_target)
    
    sim_array = np.array(simulated_spikes)
    target_array = np.array(target_spikes)

    total_error =0.0

    for target_time in target_array:
        time_diffs = np.abs(sim_array - target_time)
        min_diff = np.min(time_diffs)

        if min <= max_delay:
            total_error+=min_diff**2
        else:
            total_error+=max_delay**2

    avg_error = total_error / n_target
    return avg_error

def compute_firing_rate_loss(simulated_spikes, target_spikes, duration):
    """compute differnece in firing rates

    firing rate  = average spikes per second (Hz)


    Args:
        simulated_spikes (list): simulated spike times
        target_spikes (list): target spike times
        duration (float): Simulation duration
    """
    rate_sim = (len(simulated_spikes) / duration) *1000
    rate_target = (len(target_spikes) / duration) *1000

    loss = (rate_sim - rate_target) **2
    return loss

def compute_combined_loss(simulated_data, target_data, params, weights=None):
    """
    Compute weighted combination of multiple loss metrics

    This is the best loss function because it combines multiple aspects
    - Sub-threshold voltage dynamics (differentiable)
    - Spike count (overall excitability)
    - spike timing (temporal precision)
    - firing rate (average activity)

    Args:
        simulated_data (dict): simulated neuron data
        target_data (dict): target neuron data
        params (dict): neuron parameters
        weights (dict, optional): weights for each loss component. Defaults to None.
    """
    if weights is None:
        weights = {
            'voltage':1.0,
            'spike_count':10.0,
            'spike_timing':5.0,
            'firing_rate':2.0
        }

    #extract data
    sim_voltage = simulated_data['voltage']
    target_voltage = target_data['voltage']
    sim_spikes = simulated_data['spike_times']
    target_spikes = target_data['spike_times']
    duration = target_data['time_config']['t_total']

    #compute individual losses
    voltage_loss = compute_subthreshold_mse(sim_voltage, target_voltage, params)
    spike_count_loss = compute_spike_count_loss(sim_spikes, target_spikes)
    spike_timing_loss = compute_spike_timing_loss(sim_spikes, target_spikes)
    firing_rate_loss = compute_firing_rate_loss(sim_spikes, target_spikes, duration)

    #weight combination
    total_loss = (
        weights['voltage']*voltage_loss +
        weights['spike_count'] * spike_count_loss +
        weights['spike_timing'] * spike_timing_loss +
        weights['firing_rate'] * firing_rate_loss
    )

    result = {
        'total':total_loss,
        'components': {
            'voltage_mse':voltage_loss,
            'spike_count':spike_count_loss,
            'spike_timing':spike_timing_loss,
            'firing_rate':firing_rate_loss
        },
        'weights': weights
    }
    return result

def print_loss_comparison(loss_result):
    """
    Print detailed loss information in readable format.
    
    Args:
        loss_result (dict): Result from compute_combined_loss()
    """
    print("\n" + "="*60)
    print("LOSS COMPARISON")
    print("="*60)
    
    components = loss_result['components']
    weights = loss_result['weights']
    
    print("\nIndividual Loss Components:")
    print(f"  Voltage MSE (sub-threshold):  {components['voltage_mse']:.4f} mV²")
    print(f"  Spike Count Error:            {components['spike_count']:.2f} spikes")
    print(f"  Spike Timing Error:           {components['spike_timing']:.4f} ms²")
    print(f"  Firing Rate Error:            {components['firing_rate']:.4f} Hz²")
    
    print("\nWeights Used:")
    print(f"  Voltage:      {weights['voltage']:.1f}")
    print(f"  Spike Count:  {weights['spike_count']:.1f}")
    print(f"  Spike Timing: {weights['spike_timing']:.1f}")
    print(f"  Firing Rate:  {weights['firing_rate']:.1f}")
    
    print("\nWeighted Contributions:")
    print(f"  Voltage:      {weights['voltage'] * components['voltage_mse']:.4f}")
    print(f"  Spike Count:  {weights['spike_count'] * components['spike_count']:.4f}")
    print(f"  Spike Timing: {weights['spike_timing'] * components['spike_timing']:.4f}")
    print(f"  Firing Rate:  {weights['firing_rate'] * components['firing_rate']:.4f}")
    
    print(f"\n{'='*30}")
    print(f"TOTAL LOSS: {loss_result['total']:.4f}")
    print(f"{'='*30}\n")

def main():
    """
    Demonstrate all loss functions.
    """
    from src.layer1_parameters import get_default_parameters
    from src.layer1_input import create_constant_inputs
    from src.layer1_time_representation import create_time_configuration
    from src.layer2_full_simulation import simulate_neuron_euler
    from src.layer3_target_data import generate_target_data
    
    print("="*60)
    print("LOSS FUNCTIONS DEMONSTRATION")
    print("="*60)
    
    # Setup
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    current = create_constant_inputs(time, amplitude=18.0)
    
    # Generate target data (with "true" parameters)
    print("\n📊 Generating target data...")
    target_data = generate_target_data(current, time_config, noise_level=0.0)
    
    print(f"Target has {len(target_data['spike_times'])} spikes")
    print(f"True tau: {target_data['params']['tau']} ms")
    print(f"True threshold: {target_data['params']['v_threshold']} mV")
    
    # Example 1: Perfect match (simulate with same parameters)
    print("\n" + "="*60)
    print("EXAMPLE 1: Perfect Match (Same Parameters)")
    print("="*60)
    
    sim_voltage_perfect, sim_spikes_perfect = simulate_neuron_euler(
        target_data['params'],  # Use same parameters!
        time_config,
        current,
        v_initial=target_data['params']['v_rest']
    )
    
    simulated_perfect = {
        'voltage': sim_voltage_perfect,
        'spike_times': sim_spikes_perfect,
        'time_config': time_config
    }
    
    loss_perfect = compute_combined_loss(simulated_perfect, target_data, 
                                         target_data['params'])
    print_loss_comparison(loss_perfect)
    
    # Example 2: Slightly wrong parameters
    print("\n" + "="*60)
    print("EXAMPLE 2: Slightly Wrong Parameters")
    print("="*60)
    
    wrong_params = target_data['params'].copy()
    wrong_params['tau'] = 20.0  # True is 25.0
    wrong_params['v_threshold'] = -55.0  # True is -52.0
    
    print(f"Using wrong tau: {wrong_params['tau']} (true: {target_data['params']['tau']})")
    print(f"Using wrong threshold: {wrong_params['v_threshold']} (true: {target_data['params']['v_threshold']})")
    
    sim_voltage_wrong, sim_spikes_wrong = simulate_neuron_euler(
        wrong_params,
        time_config,
        current,
        v_initial=wrong_params['v_rest']
    )
    
    simulated_wrong = {
        'voltage': sim_voltage_wrong,
        'spike_times': sim_spikes_wrong,
        'time_config': time_config
    }
    
    loss_wrong = compute_combined_loss(simulated_wrong, target_data, 
                                       wrong_params)
    print_loss_comparison(loss_wrong)
    
    # Example 3: Very wrong parameters
    print("\n" + "="*60)
    print("EXAMPLE 3: Very Wrong Parameters")
    print("="*60)
    
    very_wrong_params = target_data['params'].copy()
    very_wrong_params['tau'] = 15.0  # Much too small
    very_wrong_params['v_threshold'] = -60.0  # Much too low
    
    print(f"Using very wrong tau: {very_wrong_params['tau']} (true: {target_data['params']['tau']})")
    print(f"Using very wrong threshold: {very_wrong_params['v_threshold']} (true: {target_data['params']['v_threshold']})")
    
    sim_voltage_very_wrong, sim_spikes_very_wrong = simulate_neuron_euler(
        very_wrong_params,
        time_config,
        current,
        v_initial=very_wrong_params['v_rest']
    )
    
    simulated_very_wrong = {
        'voltage': sim_voltage_very_wrong,
        'spike_times': sim_spikes_very_wrong,
        'time_config': time_config
    }
    
    loss_very_wrong = compute_combined_loss(simulated_very_wrong, target_data, 
                                            very_wrong_params)
    print_loss_comparison(loss_very_wrong)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Loss vs Parameter Error")
    print("="*60)
    print(f"\nPerfect match:      Total Loss = {loss_perfect['total']:.4f}")
    print(f"Slightly wrong:     Total Loss = {loss_wrong['total']:.4f}")
    print(f"Very wrong:         Total Loss = {loss_very_wrong['total']:.4f}")
    
    print("\n✅ Key Insight:")
    print("   Worse parameters → Higher loss")
    print("   This is what guides learning in Layer 5!")
    
    # Test individual loss functions
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL LOSS FUNCTIONS")
    print("="*60)
    
    print("\n1. Voltage MSE:")
    v_mse = compute_voltage_mse(simulated_wrong['voltage'], target_data['voltage'])
    print(f"   Full voltage MSE: {v_mse:.4f} mV²")
    
    print("\n2. Sub-threshold MSE:")
    sub_mse = compute_subthreshold_mse(simulated_wrong['voltage'], 
                                       target_data['voltage'], 
                                       wrong_params)
    print(f"   Sub-threshold MSE: {sub_mse:.4f} mV²")
    print(f"   (Excludes spike resets - more suitable for gradients!)")
    
    print("\n3. Spike Count Loss:")
    sc_loss = compute_spike_count_loss(simulated_wrong['spike_times'], 
                                       target_data['spike_times'])
    print(f"   Target spikes: {len(target_data['spike_times'])}")
    print(f"   Simulated spikes: {len(simulated_wrong['spike_times'])}")
    print(f"   Difference: {sc_loss:.0f} spikes")
    
    print("\n4. Spike Timing Loss:")
    st_loss = compute_spike_timing_loss(simulated_wrong['spike_times'], 
                                        target_data['spike_times'])
    print(f"   Average timing error: {st_loss:.4f} ms²")
    
    print("\n5. Firing Rate Loss:")
    fr_loss = compute_firing_rate_loss(simulated_wrong['spike_times'], 
                                       target_data['spike_times'],
                                       time_config['t_total'])
    target_rate = (len(target_data['spike_times']) / time_config['t_total']) * 1000
    sim_rate = (len(simulated_wrong['spike_times']) / time_config['t_total']) * 1000
    print(f"   Target rate: {target_rate:.2f} Hz")
    print(f"   Simulated rate: {sim_rate:.2f} Hz")
    print(f"   Squared difference: {fr_loss:.4f} Hz²")


if __name__ == "__main__":
    main()