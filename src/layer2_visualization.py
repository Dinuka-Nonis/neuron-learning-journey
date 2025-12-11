import numpy as np
import matplotlib.pyplot as plt

def plot_voltage_trace(time, voltage, spike_times, params, title="Neuron Voltage Trace"):
    """
    Plot voltage trace with spikes and reference lines

    shows how voltage evolves over time and whe spikes occur

    Args:
        time (np.ndarray): time array
        voltage (np.ndarray): voltage array
        spike_times (list): times when spikes occurred
        params (dict): neuron parameters (for reference lines)
        title (str): plot tilte.
    """
    plt.figure(figsize=(12, 6))

    #plot voltage trace
    plt.plot(time, voltage, 'b-', linewidth=2, label='Voltage')

    #add reference lines
    plt.axhline(y=params['v_rest'], color='blue', linestyle='--', 
                linewidth=1.5, alpha=0.5, label='V_rest')
    plt.axhline(y=params['v_threshold'], color='red', linestyle='--', 
                linewidth=1.5, alpha=0.5, label='V_threshold')
    plt.axhline(y=params['v_reset'], color='green', linestyle='--', 
                linewidth=1.5, alpha=0.5, label='V_reset')
    
    # Mark spike times with vertical lines
    for spike_time in spike_times:
        plt.axvline(x=spike_time, color='red', linestyle=':', 
                   linewidth=1, alpha=0.7)
    
    # Add spike markers at top of plot
    if len(spike_times) > 0:
        y_top = params['v_threshold'] + 5  # Slightly above threshold
        plt.plot(spike_times, [y_top] * len(spike_times), 
                'r^', markersize=10, label='Spikes')
    
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Voltage (mV)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_voltage_and_current(time, voltage, current, spike_times, params, 
                             title="Neuron Response to Input"):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top subplot: Voltage
    ax1.plot(time, voltage, 'b-', linewidth=2, label='Voltage')
    ax1.axhline(y=params['v_rest'], color='blue', linestyle='--', 
                linewidth=1.5, alpha=0.5, label='V_rest')
    ax1.axhline(y=params['v_threshold'], color='red', linestyle='--', 
                linewidth=1.5, alpha=0.5, label='V_threshold')
    ax1.axhline(y=params['v_reset'], color='green', linestyle='--', 
                linewidth=1.5, alpha=0.5, label='V_reset')
    
    # Mark spikes
    for spike_time in spike_times:
        ax1.axvline(x=spike_time, color='red', linestyle=':', 
                   linewidth=1, alpha=0.7)
    
    if len(spike_times) > 0:
        y_top = params['v_threshold'] + 5
        ax1.plot(spike_times, [y_top] * len(spike_times), 
                'r^', markersize=10, label='Spikes')
    
    ax1.set_ylabel('Voltage (mV)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: Input current
    ax2.plot(time, current, 'orange', linewidth=2, label='Input Current')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(time, 0, current, alpha=0.3, color='orange')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Current (a.u.)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_multiple_simulations(simulations, params, title="Comparison of Simulations"):
    """
    Plot multiple simulations for comparison.
    
    Each simulation is shown in its own subplot, making it easy
    to compare different inputs or parameters.
    
    Args:
        simulations (list): List of dicts, each containing:
            - 'time': time array
            - 'voltage': voltage array
            - 'spike_times': list of spike times
            - 'label': description of this simulation
        params (dict): Neuron parameters
        title (str): Overall title
    """
    n_sims = len(simulations)
    fig, axes = plt.subplots(n_sims, 1, figsize=(12, 4*n_sims), sharex=True)
    
    # Handle single subplot case
    if n_sims == 1:
        axes = [axes]
    
    for idx, (ax, sim) in enumerate(zip(axes, simulations)):
        time = sim['time']
        voltage = sim['voltage']
        spike_times = sim['spike_times']
        label = sim.get('label', f'Simulation {idx+1}')
        
        # Plot voltage
        ax.plot(time, voltage, 'b-', linewidth=2)
        
        # Reference lines
        ax.axhline(y=params['v_rest'], color='blue', linestyle='--', 
                  linewidth=1.5, alpha=0.5)
        ax.axhline(y=params['v_threshold'], color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.5)
        
        # Mark spikes
        for spike_time in spike_times:
            ax.axvline(x=spike_time, color='red', linestyle=':', 
                      linewidth=1, alpha=0.7)
        
        if len(spike_times) > 0:
            y_top = params['v_threshold'] + 5
            ax.plot(spike_times, [y_top] * len(spike_times), 
                   'r^', markersize=8)
        
        ax.set_ylabel('Voltage (mV)', fontsize=11)
        ax.set_title(f"{label} ({len(spike_times)} spikes)", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()
def main():
    """
    Demonstrate all visualization functions.
    """
    import sys
    sys.path.append('.')
    from src.layer1_parameters import get_default_parameters
    from src.layer1_input import create_constant_inputs, create_pulse_input
    from src.layer1_time_representation import create_time_configuration
    from src.layer2_full_simulation import simulate_neuron_euler
    
    # Setup
    params = get_default_parameters()
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    
    print("="*60)
    print("VISUALIZATION EXAMPLES")
    print("="*60)
    
    # Example 1: Basic voltage plot
    print("\n📊 Example 1: Basic Voltage Trace")
    current1 = create_constant_inputs(time, amplitude=18.0)
    voltage1, spikes1 = simulate_neuron_euler(params, time_config, current1, 
                                              v_initial=params['v_rest'])
    plot_voltage_trace(time, voltage1, spikes1, params, 
                      "Example 1: Constant Strong Input")
    
    # Example 2: Voltage + Current
    print("\n📊 Example 2: Voltage and Current Together")
    current2 = create_pulse_input(time, start_time=20.0, end_time=60.0, amplitude=20.0)
    voltage2, spikes2 = simulate_neuron_euler(params, time_config, current2, 
                                              v_initial=params['v_rest'])
    plot_voltage_and_current(time, voltage2, current2, spikes2, params,
                            "Example 2: Pulse Input Response")
    
    # Example 3: Multiple simulations
    print("\n📊 Example 3: Comparing Different Input Strengths")
    simulations = []
    
    for amplitude in [10, 15, 20, 25]:
        current = create_constant_inputs(time, amplitude=amplitude)
        voltage, spikes = simulate_neuron_euler(params, time_config, current, 
                                               v_initial=params['v_rest'])
        simulations.append({
            'time': time,
            'voltage': voltage,
            'spike_times': spikes,
            'label': f'Input = {amplitude}'
        })
    
    plot_multiple_simulations(simulations, params, 
                             "Comparison: Effect of Input Strength")
    
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()