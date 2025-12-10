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
        