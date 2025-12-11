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