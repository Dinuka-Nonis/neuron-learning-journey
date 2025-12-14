import numpy as np
import sys
sys.path.append('.')

def compute_voltage_mse(simulated_voltage, target_voltage):
    """
    Compute Mean Squared Error between voltage traces.
    
    MSE = (1/N) × Σ(v_sim - v_target)²
    """
    assert len(simulated_voltage) == len(target_voltage), \
        "Voltage traces must have same length!"
    
    squared_diff = (simulated_voltage - target_voltage)**2
    mse = np.mean(squared_diff)
    return mse


def compute_subthreshold_mse(simulated_voltage, target_voltage, params):
    """
    Compute MSE only on differentiable parts - excluding spikes.
    
    IMPORTANT: This is the most useful loss for gradient-based learning
    because it excludes the discontinuous spike resets.
    """
    v_reset = params['v_reset']
    
    # Create mask - true where voltage is not at reset
    sim_mask = simulated_voltage != v_reset
    target_mask = target_voltage != v_reset
    
    # Only include points that are sub-threshold in both traces
    combined_mask = sim_mask & target_mask
    
    # Extract sub-threshold voltages
    sim_subthreshold = simulated_voltage[combined_mask]
    target_subthreshold = target_voltage[combined_mask]
    
    if len(sim_subthreshold) == 0:
        return 0.0
    
    squared_diff = (sim_subthreshold - target_subthreshold)**2
    mse = np.mean(squared_diff)
    
    return mse


def compute_spike_count_loss(simulated_spikes, target_spikes):
    """
    Compute difference in number of spikes.
    """
    n_sim = len(simulated_spikes)
    n_target = len(target_spikes)
    
    loss = abs(n_sim - n_target)
    
    return float(loss)


def compute_spike_timing_loss(simulated_spikes, target_spikes, max_delay=10.0):
    """
    Compute spike timing mismatch using simplified matching.
    """
    n_sim = len(simulated_spikes)
    n_target = len(target_spikes)
    
    if n_sim == 0 and n_target == 0:
        return 0.0
    
    if n_sim == 0 or n_target == 0:
        return max_delay**2 * max(n_sim, n_target)
    
    sim_array = np.array(simulated_spikes)
    target_array = np.array(target_spikes)
    
    total_error = 0.0
    
    for target_time in target_array:
        time_diffs = np.abs(sim_array - target_time)
        min_diff = np.min(time_diffs)
        
        if min_diff <= max_delay:
            total_error += min_diff**2
        else:
            total_error += max_delay**2
    
    avg_error = total_error / n_target
    return avg_error


def compute_firing_rate_loss(simulated_spikes, target_spikes, duration):
    """
    Compute difference in firing rates.
    
    Firing rate = average spikes per second (Hz)
    """
    rate_sim = (len(simulated_spikes) / duration) * 1000
    rate_target = (len(target_spikes) / duration) * 1000
    
    loss = (rate_sim - rate_target)**2
    return loss


def compute_combined_loss(simulated_data, target_data, params, weights=None):
    """
    Compute weighted combination of multiple loss metrics.
    
    FIXED VERSION with better default weights:
    - Emphasizes sub-threshold voltage dynamics (differentiable)
    - Balances spike-related losses
    - More suitable for gradient descent
    
    Args:
        simulated_data (dict): Simulated neuron data
        target_data (dict): Target neuron data
        params (dict): Neuron parameters
        weights (dict, optional): Weights for each loss component
        
    Returns:
        dict: Total loss and component breakdown
    """
    
    if weights is None:
        # FIXED: Better balanced weights
        weights = {
            'voltage': 10.0,      # Emphasize voltage matching (differentiable!)
            'spike_count': 5.0,   # Moderate spike count importance
            'spike_timing': 2.0,  # Less emphasis on exact timing
            'firing_rate': 1.0    # Least emphasis (captured by spike_count)
        }
    
    # Extract data
    sim_voltage = simulated_data['voltage']
    target_voltage = target_data['voltage']
    sim_spikes = simulated_data['spike_times']
    target_spikes = target_data['spike_times']
    duration = target_data['time_config']['t_total']
    
    # Compute individual losses
    voltage_loss = compute_subthreshold_mse(sim_voltage, target_voltage, params)
    spike_count_loss = compute_spike_count_loss(sim_spikes, target_spikes)
    spike_timing_loss = compute_spike_timing_loss(sim_spikes, target_spikes)
    firing_rate_loss = compute_firing_rate_loss(sim_spikes, target_spikes, duration)
    
    # Weighted combination
    total_loss = (
        weights['voltage'] * voltage_loss +
        weights['spike_count'] * spike_count_loss +
        weights['spike_timing'] * spike_timing_loss +
        weights['firing_rate'] * firing_rate_loss
    )
    
    result = {
        'total': total_loss,
        'components': {
            'voltage_mse': voltage_loss,
            'spike_count': spike_count_loss,
            'spike_timing': spike_timing_loss,
            'firing_rate': firing_rate_loss
        },
        'weights': weights
    }
    
    return result


def print_loss_comparison(loss_result):
    """
    Print detailed loss information in readable format.
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


def get_loss_weights_profile(profile='balanced'):
    """
    Get pre-configured loss weight profiles for different optimization goals.
    
    ADDED: Provides different weight configurations for different scenarios.
    
    Args:
        profile (str): Weight profile name
            - 'balanced': Default balanced weights
            - 'voltage_focused': Emphasize voltage matching
            - 'spike_focused': Emphasize spike matching
            - 'timing_focused': Emphasize spike timing
            
    Returns:
        dict: Weight configuration
    """
    profiles = {
        'balanced': {
            'voltage': 10.0,
            'spike_count': 5.0,
            'spike_timing': 2.0,
            'firing_rate': 1.0
        },
        'voltage_focused': {
            'voltage': 20.0,      # Much higher weight
            'spike_count': 2.0,
            'spike_timing': 1.0,
            'firing_rate': 0.5
        },
        'spike_focused': {
            'voltage': 5.0,
            'spike_count': 10.0,   # Much higher weight
            'spike_timing': 5.0,
            'firing_rate': 3.0
        },
        'timing_focused': {
            'voltage': 5.0,
            'spike_count': 3.0,
            'spike_timing': 10.0,  # Much higher weight
            'firing_rate': 2.0
        }
    }
    
    if profile not in profiles:
        print(f"Warning: Unknown profile '{profile}', using 'balanced'")
        profile = 'balanced'
    
    return profiles[profile]


def main():
    """
    Demonstrate loss functions with FIXED weights.
    """
    from src.layer1_parameters import get_default_parameters
    from src.layer1_input import create_constant_inputs
    from src.layer1_time_representation import create_time_configuration
    from src.layer2_full_simulation import simulate_neuron_euler
    from src.layer3_target_data import generate_target_data
    
    print("="*60)
    print("LOSS FUNCTIONS (FIXED WEIGHTS)")
    print("="*60)
    print("\nKey Fix: Better balanced weights")
    print("- Voltage weight increased (10.0)")
    print("- Spike count weight decreased (5.0)")
    print("- More suitable for gradient descent\n")
    
    # Setup
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    current = create_constant_inputs(time, amplitude=20.0)
    
    # Generate target
    print("📊 Generating target data...")
    target_data = generate_target_data(current, time_config, noise_level=0.0)
    
    print(f"Target has {len(target_data['spike_times'])} spikes")
    print(f"True tau: {target_data['params']['tau']} ms")
    print(f"True threshold: {target_data['params']['v_threshold']} mV")
    
    # Example 1: Perfect match
    print("\n" + "="*60)
    print("EXAMPLE 1: Perfect Match")
    print("="*60)
    
    sim_voltage_perfect, sim_spikes_perfect = simulate_neuron_euler(
        target_data['params'],
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
    
    # Example 2: Slightly wrong
    print("\n" + "="*60)
    print("EXAMPLE 2: Slightly Wrong Parameters")
    print("="*60)
    
    wrong_params = target_data['params'].copy()
    wrong_params['tau'] = 20.0
    wrong_params['v_threshold'] = -55.0
    
    print(f"Using tau: {wrong_params['tau']} (true: {target_data['params']['tau']})")
    print(f"Using threshold: {wrong_params['v_threshold']} (true: {target_data['params']['v_threshold']})")
    
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
    
    loss_wrong = compute_combined_loss(simulated_wrong, target_data, wrong_params)
    print_loss_comparison(loss_wrong)
    
    # Example 3: Different weight profiles
    print("\n" + "="*60)
    print("EXAMPLE 3: Different Weight Profiles")
    print("="*60)
    
    profiles = ['balanced', 'voltage_focused', 'spike_focused', 'timing_focused']
    
    for profile in profiles:
        weights = get_loss_weights_profile(profile)
        loss_result = compute_combined_loss(simulated_wrong, target_data, 
                                           wrong_params, weights=weights)
        
        print(f"\n{profile.upper()}:")
        print(f"  Total loss: {loss_result['total']:.4f}")
        print(f"  Voltage contribution: {weights['voltage'] * loss_result['components']['voltage_mse']:.4f}")
        print(f"  Spike contribution: {weights['spike_count'] * loss_result['components']['spike_count']:.4f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("\nFor gradient descent, use 'voltage_focused' or 'balanced'")
    print("because sub-threshold voltage MSE is:")
    print("  ✓ Differentiable everywhere")
    print("  ✓ Smooth loss landscape")
    print("  ✓ Provides useful gradients")
    print("\nSpike-based losses are:")
    print("  ✗ Discontinuous (spikes are all-or-nothing)")
    print("  ✗ Non-differentiable")
    print("  ✓ Good for final evaluation")
    
    print("\n✅ LOSS FUNCTIONS WITH BETTER WEIGHTS!")


if __name__ == "__main__":
    main()