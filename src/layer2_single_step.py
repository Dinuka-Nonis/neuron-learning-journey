"""
Layer 2.1: Single Time Step Simulation
=======================================
This module implements the core time-stepping logic for neuron simulation.

Using Forward Euler method: V_next = V_current + dV/dt * dt
"""
import numpy as np

def simulate_single_step_euler(V_current, I_current, params, dt, ):
    """
    Simulate one time step using forward euler method.
    This is the core of simulation. everything is build on this!
    The LIF equation: dV/dt = (-(V - v_rest) + I ) / tau
    Euler approximation: V_next = V_current + dV/dt * dt

    Args:
        V_current (float): Current voltage in mV
        I_current (float): Input current at this moment
        params (dict): Neuron parameters
        dt (float): Time step in ms

    Returns:
        float: Next voltage value( before spike check)
    """

    tau = params['tau']
    v_rest = params['v_rest']

    #step 1: Calculate leak (restoring toward rest)
    leak = -(V_current - v_rest)

    #step 2: Add input current
    total_drive = leak + I_current

    #step 3: Divide by time constant (controls speed)
    dV_dt= total_drive/tau

    #step 4: Scale by time step and update
    delta_V = dV_dt * dt
    V_next = V_current + delta_V

    return V_next

def check_and_handle_spike(V, params):
    """
    Check if voltage crossed threshold and handle spike resest.

    when neuron spikes:
    1. Voltage is reset to v_reset (hyperpolarization)
    2. We record that a spike occured

    This implements the "Integrate_and_fire" mechanism:
    - Integrate: voltage accumulates input
    - Fire: when threshold crossed, spike!

    Args:
        V (float): Current voltage in mV
        params (dict): Neuron parameters
    """

    v_threshold = params['v_threshold']
    v_reset = params['v_reset']

    #check if voltage crossed threshold
    if V >= v_threshold:
        #SPIKE !! reset voltage
        v_after_spike = v_reset
        spike_occured = True
        return v_after_spike, spike_occured
    else:
        #no spike, voltage unchanged
        spike_occured = False
        return V, spike_occured
    
def simulate_single_step_with_spike(V_current, I_current, params, dt):
    """
    Simulate one complete time step including spike detection

    This is the complete single step function that combines:
    1.Voltage dynamics (Euler integration)
    2. Spike detection and reset

    Order of operations
    1. Update voltage based on current state
    2. Check if updated voltage crossed threshold
    3. Reset if spike occured

    Args:
        V_current (float): Current in mV
        I_current (float): Input current at the moment
        params (dict): Neuron parameters
        dt (float): TIme step in ms
    """
    #step1 : Calculate voltage change (continous dynamics)
    V_next = simulate_single_step_euler(V_current, I_current, params, dt)

    #step2 : check for spike and handle reset (discrete event)
    V_next, spike_occurred = check_and_handle_spike(V_next, params)

    return V_next, spike_occurred

def main():
    """
    Demonstrate single time step simulation with examples.
    """
    import sys
    sys.path.append('.')
    from src.layer1_parameters import get_default_parameters
    
    params = get_default_parameters()
    dt = 0.1
    
    print("="*60)
    print("SINGLE TIME STEP EXAMPLES")
    print("="*60)
    
    # Example 1: Sub-threshold (FIXED - will actually change now)
    print("\nExample 1: Sub-threshold Response")
    print("-" * 40)
    V_current = -65.0
    I_current = 10.0  # Changed from 5.0 to 10.0!
    
    print(f"Starting voltage: {V_current:.2f} mV")
    print(f"Input current: {I_current:.1f}")
    
    V_next, spiked = simulate_single_step_with_spike(V_current, I_current, params, dt)
    
    print(f"Next voltage: {V_next:.6f} mV")
    print(f"Spike occurred: {spiked}")
    print(f"Change: {V_next - V_current:.6f} mV")
    
    # Example 2: At rest
    print("\nExample 2: At Rest (No Input)")
    print("-" * 40)
    V_current = params['v_rest']
    I_current = 0.0
    
    print(f"Starting voltage: {V_current:.2f} mV")
    print(f"Input current: {I_current:.1f}")
    
    V_next, spiked = simulate_single_step_with_spike(V_current, I_current, params, dt )
    
    print(f"Next voltage: {V_next:.6f} mV")
    print(f"Spike occurred: {spiked}")
    print(f"Change: {V_next - V_current:.6f} mV")
    
    # Example 3: Supra-threshold (FIXED - will actually spike now)
    print("\nExample 3: Strong input, near threshold")
    print("-" * 40)
    V_current = -55.01  # Changed from -56 to -55.5 (closer to threshold)
    I_current = 30.0   # Changed from 20 to 30 (much stronger!)
    
    print(f"Starting voltage: {V_current:.2f} mV")
    print(f"Input current: {I_current:.1f}")
    print(f"Threshold: {params['v_threshold']:.2f} mV")
    
    V_next, spiked = simulate_single_step_with_spike(V_current, I_current, params, dt )
    
    print(f"Next voltage: {V_next:.6f} mV")
    print(f"Spike occurred: {spiked} ")
    if spiked:
        print(f"Reset to: {params['v_reset']:.2f} mV")
    
    # Example 4: Sequence
    print("\nExample 4: Sequence of 5 Steps")
    print("-" * 40)
    V = -70.0
    I = 10.0
    
    print(f"Initial voltage: {V:.2f} mV")
    print(f"Constant input: {I:.1f}")
    print("\nStep-by-step evolution:")
    
    for step in range(5):
        V, spiked = simulate_single_step_with_spike(V, I, params, dt ) 
        spike_marker = " SPIKE!" if spiked else ""
        print(f"  Step {step+1}: V = {V:.3f} mV {spike_marker}")


if __name__ == "__main__":
    main()