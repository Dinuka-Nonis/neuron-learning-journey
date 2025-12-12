import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('.')
from src.layer1_input import create_constant_inputs
from src.layer1_time_representation import create_time_configuration
from src.layer2_full_simulation import simulate_neuron_euler
from src.layer3_target_data import generate_target_data
from src.layer3_loss_functions import compute_combined_loss

def compute_gradient_finite_diff(param_name, params, target_data, current, time_config, h=0.01):
    """
    Compute gradient of loss w.r.t. ONE paraameter using finite differences

    this is numerical differentiation : 
        ∂loss/∂param ≈ (loss(param+h) - loss(param)) / h


    Args:
        param_name (str): Name of parameter
        params (dict): current parameter values
        target_data (dict): target data to match
        current (np.ndarray): input current
        time_config (dict): time configuration
        h (float, optional): step sixe for finite difference . Defaults to 0.01.
    """

    #step 1: compute loss at original parameters
    v_initial = params['v_rest']
    voltage_original, spikes_original = simulate_neuron_euler(
        params, time_config, current, v_initial
    )
    simulated_original = {
        'voltage':voltage_original,
        'spike_times':spikes_original,
        'time_config':time_config
    }

    loss_result_original = compute_combined_loss(
        simulated_original, target_data, params
    )
    loss_original = loss_result_original['total']

    #step2: perturb the parameter slightly

    params_pertubed = params.copy()
    original_value = params[param_name]
    params_pertubed[param_name] = original_value + h

    #step3 : compute loss at perbuted parameters
    voltage_perturbed, spikes_perturbed = simulate_neuron_euler(
        params_pertubed, time_config, current, v_initial
    )

    simulated_perturbed = {
        'voltage':voltage_perturbed,
        'spike_times':spikes_perturbed,
        'time_config':time_config
    }

    loss_result_perturbed = compute_combined_loss(simulated_perturbed, target_data, params_pertubed)

    loss_perturbed = loss_result_perturbed['total']

    #step4 : compute gradient 
    gradient = (loss_perturbed - loss_original)/h

    result = {
        'gradient':gradient,
        'loss_original':loss_original,
        'loss_perturbed':loss_perturbed,
        'param_original':original_value,
        'param_perturbed':original_value+h
    }
    return result

def compute_all_gradients_finite_diff(params, target_data, current,time_config, h=0.01, params_to_optimize =None):
    """
    compute gradients for all parameters
    this calls compute_gradient_finite_diff() for each parameter

    Args:
        params (dict): current parameter values
        target_data (dicct): target data to match
        current (np.ndarray): input current
        time_config (dict): time configuration
        h (float, optional): step size for finite differences. Defaults to 0.01.
        params_to_optimize (list, optional): which parameters tocompute gradients for. Defaults to None.
    """
    #default : optimize all parameters
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_rest', 'v_threshold', 'v_reset']

    #storage
    gradients = {}
    details = {}

    #compute gradient for each parameter
    print("\nComputing gradients...")
    for param_name in params_to_optimize:
        print(f" Computing ∂loss/∂{param_name}...", end=' ')

        result = compute_gradient_finite_diff(
            param_name, params, target_data, current, time_config, h
        )

        gradients[param_name] = result['gradient']
        details[param_name] = result

        print(f"✓ gradient = {result['gradient']:.4f}")

    loss_original = details[params_to_optimize[0]]['loss_original']

    result = {
        'gradients':gradients,
        'loss':loss_original,
        'details':details
    }

    return result

def verify_gradient_direction(param_name , params , gradient , target_data, current, time_config, step_size = 0.1):
    """
    verify that gradient points in right direction

    - takes a step in gradient direction and checks if loss increase
    - takes a step in opposite direction and checks if loss decreases

    this is a sanity check that our gradient computation is correct!

    Args:
        param_name (str): paramters name
        params (dict): current parameters
        gradient (float): computed gradients
        target_data (dict): target data
        current (np.ndarray): input current
        time_config (dict): time configuration
        step_size (float, optional): how large a step to take for verification. Defaults to 0.1.
    """

     # Original loss
    v_initial = params['v_rest']
    voltage_orig, spikes_orig = simulate_neuron_euler(
        params, time_config, current, v_initial
    )
    simulated_orig = {
        'voltage': voltage_orig,
        'spike_times': spikes_orig,
        'time_config': time_config
    }
    loss_orig = compute_combined_loss(simulated_orig, target_data, params)['total']

    #step in gradient should increase loss
    params_plus = params.copy()
    params_plus[param_name] = params[param_name] + step_size

    voltage_plus, spikes_plus = simulate_neuron_euler(
        params_plus, time_config, current, v_initial
    )
    simulated_plus = {
        'voltage': voltage_plus,
        'spike_times': spikes_plus,
        'time_config': time_config
    }
    loss_plus = compute_combined_loss(simulated_plus, target_data, params_plus)['total']
    
    # Step in opposite direction (should decrease loss if gradient is positive)
    params_minus = params.copy()
    params_minus[param_name] = params[param_name] - step_size
    
    voltage_minus, spikes_minus = simulate_neuron_euler(
        params_minus, time_config, current, v_initial
    )
    simulated_minus = {
        'voltage': voltage_minus,
        'spike_times': spikes_minus,
        'time_config': time_config
    }
    loss_minus = compute_combined_loss(simulated_minus, target_data, params_minus)['total']
    
    # Check if gradient direction is correct
    if gradient > 0:
        # Positive gradient: loss should increase when we increase param
        correct = loss_plus > loss_orig
    elif gradient < 0:
        # Negative gradient: loss should decrease when we increase param
        correct = loss_plus < loss_orig
    else:
        # Zero gradient: loss shouldn't change much
        correct = abs(loss_plus - loss_orig) < 0.01
    
    result = {
        'param_name': param_name,
        'gradient': gradient,
        'loss_original': loss_orig,
        'loss_plus': loss_plus,
        'loss_minus': loss_minus,
        'correct': correct
    }
    
    return result