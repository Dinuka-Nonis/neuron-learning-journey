import numpy as np
import matplotlib.pyplot as plt

def create_time_grid(dt, t_total):
    """
    Create a time grid for simulation.

    Args:
        dt (float): Time to step in milliseconds
        t_total (float): Total simulation time in milliseconds

    Returns:
        np.ndarray: Array of time points
    """

    time = np.arrange(0, t_total + dt, dt)
    return time

def create_constant_inputs(time, amplitude):
    """
    Create a constant input current.

    Args:
        time (np.ndarray): Time grid
        amplitude (float): Current amplitude (typically in nA or pA)

    Returns:
        np.ndarray: Current values at each time point
    """
# Used for testing if neuron spikes with steady input
# A simplest experiment
# Finding minimum current needed to spike

    current = np.ones_like(time)* amplitude   #creates an array of 1's . if time has 1000 points, current has 1000 points
    return current
# multiplied by amplitude because we want all values to be 'amplitude' , not 1

def create_pulse_input(time, start_time, end_time, amplitude):
    """
    Create a pulse input current (on during a time window).

    Args:
        time (np.ndarray): Time grid
        start_time (float): When pulse starts (ms)
        end_time (float): When pulse ends (ms)
        amplitude (float): Current amplitude during pulse

    Returns:
        np.ndarray:  Current values at each time point
    """
    current  = np.zeros_like(time)  # start with zeros  - same length as time
    pulse_indices = (time >= start_time) & (time<= end_time)  # find indices where time is in range
    #sets values to amplitude only where mask is true
    current[pulse_indices] = amplitude   # set those time to amplitude
    return current

def plot_input_current(time, current, title="Input Current"):
    """
    Plot input current over time.

    Args:
        time (np.ndarray): Time grid
        current (np.ndarray): Current values
        title (str): Plot title. Defaults to "Input Current".
    """

    plt.figure(figsize=(10, 4))
    plt.plot(time, current, linewidth=2, color='blue')
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Input Current (arbitrary units)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()