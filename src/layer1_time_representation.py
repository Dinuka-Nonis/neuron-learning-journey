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

def get_time_stats(time_config):
    """
    Calculate usefule statistics about time configuration.

    Args:
        time_config (dict): Time configuration

    Returns:
        dict: Statistics including:
            - memory_mb: approximate memory for arrays
            - steps_per_ms: temporal resolution
            - total_seconds: duration in seconds  
    """

    n_steps = time_config['n_steps']
    dt = time_config['dt']
    t_total = time_config['t_total']

    #estimate memory (each float64 = 8 bytes)
    #We'll have : time, current, voltage arrays
    bytes_per_array = n_steps*8
    total_bytes = bytes_per_array*3 #3 main arrays
    memory_mb = total_bytes/(1024*1024)

    stats = {
        'memory_mb': memory_mb,
        'steps_per_ms': 1.0 / dt,
        'total_seconds': t_total / 1000.0,
        'sampling_rate_hz': 1000.0 / dt
    }
    return stats

def main():
    """
    Demonstrate time configurations with different resolutions.
    """
    print("="*60)
    print("TIME REPRESENTATION EXAMPLES")
    print("="*60)
    
    # Example 1: Fine resolution (accurate but slow)
    print("\nExample 1: Fine Resolution")
    config_fine = create_time_configuration(dt=0.05, t_total=100.0)
    print_time_info(config_fine)
    stats_fine = get_time_stats(config_fine)
    print(f"Memory needed: {stats_fine['memory_mb']:.4f} MB")
    print(f"Steps per millisecond: {stats_fine['steps_per_ms']:.1f}")
    
    # Example 2: Standard resolution (balanced)
    print("\nExample 2: Standard Resolution")
    config_standard = create_time_configuration(dt=0.1, t_total=100.0)
    print_time_info(config_standard)
    stats_standard = get_time_stats(config_standard)
    print(f"Memory needed: {stats_standard['memory_mb']:.4f} MB")
    print(f"Steps per millisecond: {stats_standard['steps_per_ms']:.1f}")
    
    # Example 3: Coarse resolution (fast but less accurate)
    print("\nExample 3: Coarse Resolution")
    config_coarse = create_time_configuration(dt=0.5, t_total=100.0)
    print_time_info(config_coarse)
    stats_coarse = get_time_stats(config_coarse)
    print(f"Memory needed: {stats_coarse['memory_mb']:.4f} MB")
    print(f"Steps per millisecond: {stats_coarse['steps_per_ms']:.1f}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Fine uses {config_fine['n_steps'] / config_coarse['n_steps']:.1f}x more steps than coarse")
    print(f"Fine uses {stats_fine['memory_mb'] / stats_coarse['memory_mb']:.1f}x more memory than coarse")


if __name__ == "__main__":
    main()