import numpy as np

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

