"""
Layer 2.1: Single Time Step Simulation
=======================================
This module implements the core time-stepping logic for neuron simulation.

Using Forward Euler method: V_next = V_current + dV/dt * dt
"""
import numpy as np

def simulate_single_step_euler(V_current, I_current, params, dt):
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