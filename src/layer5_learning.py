import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from src.layer1_parameters import get_default_parameters
from src.layer1_input import create_constant_inputs
from src.layer1_time_representation import create_time_configuration
from src.layer2_full_simulation import simulate_neuron_euler
from src.layer3_target_data import generate_target_data
from src.layer3_loss_functions import compute_combined_loss
from src.layer4_gradients import compute_all_gradients_finite_diff

def gradient_descent(initial_params, target_data, current, time_config,
                     learning_rate = 0.1, max_iterations=100 , tolerance = 0.01,
                     params_to_optimize=None, verbose = True):
    """
    Optimize neuron parameters using gradient descent.
    
    This is THE learning algorithm - repeatedly:
    1. Simulate neuron
    2. Compute loss
    3. Compute gradients
    4. Update parameters
    
    Until loss is small or max iterations reached.
    
    Args:
        initial_params (dict): Starting parameter values
        target_data (dict): Target data to match
        current (np.ndarray): Input current
        time_config (dict): Time configuration
        learning_rate (float): Step size for updates (default: 0.1)
        max_iterations (int): Maximum number of iterations (default: 100)
        tolerance (float): Stop if loss < tolerance (default: 0.01)
        params_to_optimize (list): Which parameters to optimize (default: all)
        verbose (bool): Print progress (default: True) 
    """
    
    # Initialize
    params = initial_params.copy()
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_rest','v_threshold','v_reset']

    # History tracking
    history = {
        'loss': [],
        'params': {name: [] for name in params_to_optimize}
    }
    if verbose:
        print("\n" + "="*60)
        print("GRADIENT DESCENT OPTIMIZATION")
        print("="*60)
        print(f"\nSettings:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Optimizing: {params_to_optimize}")
        print("\nStarting optimization...\n")

    # Main Learning loop
    for iteration in range(max_iterations):
        # step1 : Simulate with current parameters (layer2)
        v_initial = params['v_rest']
        voltage, spikes = simulate_neuron_euler(
            params, time_config, current, v_initial
        )
        simulated = {
            'voltage': voltage,
            'apike_times': spikes,
            'time_config': time_config
        }
    
        # step2 : Compute loss (layer3)
        loss_result = compute_combined_loss(simulated, target_data, params)
        loss = loss_result['total']

        # Record history
        history['loss'].append(loss)
        for param_name in params_to_optimize:
            history['params'][param_name].append(params[param_name])

        # Print progress
        if verbose and (iteration % 10 == 0 or iteration == max_iterations -1):
            print(f"Iteration {iteration:3d}: Loss = {loss:.6f}")
            if iteration % 20 == 0:
                for param_name in params_to_optimize:
                    print(f"  {param_name:12s} = {params[param_name]:8.3f}")

        # Check convergence
        if loss < tolerance:
            if verbose:
                print(f"\n Converged! Loss < {tolerance} ata iteration {iteration}")
            break

        # step3 : Compute gradients (layer4)
        gradient_result = compute_all_gradients_finite_diff(
            params, target_data, current, time_config,
            h=0.01, params_to_optimize=params_to_optimize
        )
        gradients = gradient_result['gradients']

        # step4 : Update parameters 
        for param_name in params_to_optimize:
            gradient = gradients[param_name]
            params[param_name] -= learning_rate*gradient

    # Final Results
    converged = loss < tolerance

    if verbose:
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"\nFinal loss: {loss:.6f}")
        print(f"Iterations: {iteration + 1}")
        print(f"Converged: {converged}")
        print("\nFinal parameters:")
        for param_name in params_to_optimize:
            print(f"  {param_name:12s} = {params[param_name]:8.3f}")

    result = {
        'final_params':params,
        'history':history,
        'converged':converged,
        'iterations':iteration+1,
        'final_loss': loss
    }
    return result

def gradient_descent_with_momentum(initial_params, target_data, current, time_config,
                                   learning_rate=0.1, momentum = 0.9 , max_iterations = 100, 
                                   tolerance = 0.01, params_to_optimize=None, verbose = True):
    """
    Gradient descent with momentum ( accelerated optimization)

    momemntum helps by:
    -smoothing out oscillations
    -accelerating in  consistent directions
    -escaping shallow local minima

    update rule: 
        velocity = momentum * velocity - learning_rate* gradient
        params = params + velocity  

    Args:
        same as gradient_descent, plus:
        momentum (float): Momemntum coefficent (0-1, default: 0.9)
    """

    params = initial_params.copy()
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_rest', 'v_threshold', 'v_reset']
    
    # Initialize velocity (momentum term)
    velocity = {name: 0.0 for name in params_to_optimize}
    
    history = {
        'loss': [],
        'params': {name: [] for name in params_to_optimize}
    }
    
    if verbose:
        print("\n" + "="*60)
        print("GRADIENT DESCENT WITH MOMENTUM")
        print("="*60)
        print(f"\nSettings:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Momentum: {momentum}")
        print(f"  Max iterations: {max_iterations}")
        print("\nStarting optimization...\n")
    
    for iteration in range(max_iterations):
        # Simulate
        v_initial = params['v_rest']
        voltage, spikes = simulate_neuron_euler(
            params, time_config, current, v_initial
        )
        
        simulated = {
            'voltage': voltage,
            'spike_times': spikes,
            'time_config': time_config
        }
        
        # Loss
        loss_result = compute_combined_loss(simulated, target_data, params)
        loss = loss_result['total']
        
        # Record
        history['loss'].append(loss)
        for param_name in params_to_optimize:
            history['params'][param_name].append(params[param_name])
        
        # Progress
        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print(f"Iteration {iteration:3d}: Loss = {loss:.6f}")
        
        # Check convergence
        if loss < tolerance:
            if verbose:
                print(f"\n✅ Converged at iteration {iteration}")
            break
        
        # Gradients
        gradient_result = compute_all_gradients_finite_diff(
            params, target_data, current, time_config,
            h=0.01, params_to_optimize=params_to_optimize
        )
        gradients = gradient_result['gradients']
        
        # Update with momentum
        for param_name in params_to_optimize:
            # Update velocity (momentum + gradient)
            velocity[param_name] = (momentum * velocity[param_name] - 
                                   learning_rate * gradients[param_name])
            
            # Update parameter
            params[param_name] += velocity[param_name]
    
    converged = loss < tolerance
    
    if verbose:
        print("\n" + "="*60)
        print(f"Final loss: {loss:.6f} after {iteration + 1} iterations")
        print("="*60)
    
    result = {
        'final_params': params,
        'history': history,
        'converged': converged,
        'iterations': iteration + 1,
        'final_loss': loss
    }
    
    return result