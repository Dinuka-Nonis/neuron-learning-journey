def get_default_parameters():
    """
    Returns default parameters for a Leaky Integrate-and-Fire (LIF) neuron

    Returns: 
        dict: Dictionary containing neuron parameters
            - tau: membrane time constant (ms)
            - v_rest: resting potential (mV)
            -v_threshold: spike threshold (mV)
            -v_reset: reset potential after spike (mV)
    """
    parameters = {
        "tau":20.0,         # .0 makes it float - for scientific computing
        "v_rest":-70.0,
        "v_threshold":-55.0,
        "v_reset":-80.0
    }
    return parameters