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

def print_parameters(params):    # why have this function - we might want to print parameters many times
    """
    Print neuron parameters in a readable format.

    Args:
        params (dict): Dictionary of neuron parameters
    """
    print("\n" + "="*50)  # Creating a string with 50 equal signs
    print("NEURON PARAMETERS")
    print("="*50)
    print(f"Membrane time constant (tau):   {params['tau']:.1f} ms")  # {params['tau] gets th evalue from dictionary
    #:.1f formats it to a decimal place  --- eg :- 20.0 not 20.00000
    print(f"Resting potential (v_rest):    {params['v_rest']:.1f} mV")
    print(f"Spike threshold (v_threshold): {params['v_threshold']:.1f} mV")
    print(f"Reset potential (v_reset):     {params['v_reset']:.1f} mV")
    print("="*50 + "\n")

def main():
    """
    Main function to demonstrate parameter definition.
    """

    params = get_default_parameters()  # reason to store result - so we can use it multiple times

    print_parameters(params)

    print ("Individual access examples: ")
    print(f"The neuron's time constant is {params['tau']}ms")
    print(f"It will spike when voltage reaches {params['v_threshold']} mV")

if __name__ == "__main__":  # this means "Only run main() when this file is executed directly" If we import this file elsewhere, main() wont run automatically
    main()