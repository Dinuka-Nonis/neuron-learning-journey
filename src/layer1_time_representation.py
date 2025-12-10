import numpy as np

def validate_time_parameters(dt, t_total):
    """
    Validate time parameters before creating configuration.

    Args:
        dt (float): Time step in milliseconds
        t_total (float): Total simulation time in milliseconds

    Raises:
        ValueError: If parameters are invalid
    """

    if dt<= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}")
    
    if t_total <= 0:
        raise ValueError(f"Total time must be positive, got {t_total}")
    
    if dt> t_total:
        raise ValueError(f"Time step dt ({dt}) cannot be larger than total time ({t_total})")
    
    if dt < 0.01:
        print(f"WARNING : Very small time step (dt = {dt}) may result in slow simulation")

    if dt > 1.0:
        print(f"WARNING : Very large time step (dt = {dt}) may miss fast dynamics")

def create_time_configuration(dt, t_total)      :
    """
    Create a complete time configuration for simulation.

    This bundles all time-related information into one dictionary
    for easy passing to other functions

    Args:
        dt (float): Time step in milliseconds
        t_total (float): Total simulation time in milliseconds

    Returns:
        dict: Time configuration containing:
            - dt: time step
            - t_total: total time
            - time: array of time points
            - n_steps: number of time steps
    """

    #validate inputs first
    validate_time_parameters(dt, t_total)

    #create time array
    time = np.arange(0, t_total + dt, dt)

    #calculate number of steps
    n_steps = len(time)

    time_config = {
        'dt':dt,
        't_total': t_total,
        'time': time,
        'n_steps': n_steps
    }

    return time_config

def print_time_info(time_config):
    """
    Print time configuration in a readable format.

    Args:
        time_config (dict): Time configuration from create_time_configuration()
    """

    print("\n" + "="*50)
    print("TIME CONFIGURATION")
    print("="*50)
    print(f"Time step (dt):        {time_config['dt']:.4f} ms")
    print(f"Total duration:        {time_config['t_total']:.2f} ms")
    print(f"Number of steps:       {time_config['n_steps']}")
    print(f"Sampling rate:         {1000/time_config['dt']:.1f} Hz")
    print(f"Time range:            [{time_config['time'][0]:.2f}, {time_config['time'][-1]:.2f}] ms")
    print("="*50 + "\n")