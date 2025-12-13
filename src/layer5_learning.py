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
            'spike_times': spikes,
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

def plot_learning_curve(history, title="Learning Curve"):
    """
    Plot loss over iterations.
    
    Args:
        history (dict): History from gradient_descent()
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see exponential decay
    plt.tight_layout()
    plt.show()

def plot_parameter_evolution(history, true_params=None, title="Parameter Evolution"):
    """
    Plot how parameters change over iterations

    Args:
        history (dict): history from gradient_descent()
        tru_params (dict, optional): True parameters values. Defaults to None.
        title (str, optional): Plot title. Defaults to "Parameter Evolution".
    """

    params_optimized = list(history['params'].keys())
    n_params = len(params_optimized)
    
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params), sharex=True)
    
    if n_params == 1:
        axes = [axes]
    
    for i, param_name in enumerate(params_optimized):
        ax = axes[i]
        values = history['params'][param_name]
        
        ax.plot(values, 'b-', linewidth=2, label='Learned')
        
        if true_params is not None:
            true_val = true_params[param_name]
            ax.axhline(y=true_val, color='red', linestyle='--', 
                      linewidth=2, label='True value')
        
        ax.set_ylabel(param_name, fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Iteration', fontsize=12)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def compare_learned_vs_target(learned_params, target_data, current, time_config):
    """
    Compare learned parameters against target by simulating both.
    
    Args:
        learned_params (dict): Learned parameters
        target_data (dict): Target data
        current (np.ndarray): Input current
        time_config (dict): Time configuration
    """
    # Simulate with learned parameters
    v_initial = learned_params['v_rest']
    voltage_learned, spikes_learned = simulate_neuron_euler(
        learned_params, time_config, current, v_initial
    )
    
    # Extract target data
    voltage_target = target_data['voltage']
    spikes_target = target_data['spike_times']
    time = target_data['time']
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Voltage comparison
    ax1.plot(time, voltage_target, 'r-', linewidth=2, alpha=0.7, label='Target')
    ax1.plot(time, voltage_learned, 'b--', linewidth=2, alpha=0.7, label='Learned')
    
    for spike_time in spikes_target:
        ax1.axvline(x=spike_time, color='red', linestyle=':', alpha=0.3)
    for spike_time in spikes_learned:
        ax1.axvline(x=spike_time, color='blue', linestyle=':', alpha=0.3)
    
    ax1.set_ylabel('Voltage (mV)', fontsize=12)
    ax1.set_title('Learned vs Target Voltage', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Input current
    ax2.plot(time, current, 'orange', linewidth=2)
    ax2.fill_between(time, 0, current, alpha=0.3, color='orange')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Input Current', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("LEARNED VS TARGET COMPARISON")
    print("="*60)
    print(f"\nSpikes:")
    print(f"  Target:  {len(spikes_target)} spikes")
    print(f"  Learned: {len(spikes_learned)} spikes")
    print(f"  Match:   {len(spikes_target) == len(spikes_learned)}")
    
    print(f"\nVoltage MSE: {np.mean((voltage_learned - voltage_target)**2):.6f} mV²")
    
    print(f"\nParameters:")
    true_params = target_data['params']
    print(f"{'Parameter':<15} {'True':>10} {'Learned':>10} {'Error':>10}")
    print("-"*50)
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        true_val = true_params[param_name]
        learned_val = learned_params[param_name]
        error = abs(learned_val - true_val)
        print(f"{param_name:<15} {true_val:10.2f} {learned_val:10.2f} {error:10.4f}")


def main():
    """
    Complete learning demonstration - the grand finale!
    """
    print("="*60)
    print("COMPLETE LEARNING DEMONSTRATION ")
    print("="*60)
    print("\nThis brings together ALL previous layers:")
    print("  Layer 1: Parameters, input, voltage, time")
    print("  Layer 2: Forward simulation")
    print("  Layer 3: Loss functions")
    print("  Layer 4: Gradient computation")
    print("  Layer 5: Learning loop")
    
    # Setup
    print("\nSetup...")
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    current = create_constant_inputs(time, amplitude=18.0)
    
    # Generate target
    print("Generating target data...")
    target_data = generate_target_data(current, time_config, noise_level=0.0)
    
    true_params = target_data['params']
    print(f"\nTrue (hidden) parameters to recover:")
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param_name:12s} = {true_params[param_name]:.2f}")
    
    # Starting point
    initial_params = get_default_parameters()
    print(f"\nStarting parameters (initial guess):")
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param_name:12s} = {initial_params[param_name]:.2f}")
    
    # LEARNING!
    print("\n" + "="*60)
    print("Starting Learning...")
    print("="*60)
    
    result = gradient_descent(
        initial_params, target_data, current, time_config,
        learning_rate=0.5,
        max_iterations=50,
        tolerance=0.01,
        params_to_optimize=['tau', 'v_threshold'],  # Learn just these two
        verbose=True
    )
    
    # Results
    learned_params = result['final_params']
    
    print("\n" + "="*60)
    print("VISUALIZING RESULTS")
    print("="*60)
    
    # Plot 1: Learning curve
    print("\n1. Loss over time (learning curve)...")
    plot_learning_curve(result['history'], "Loss During Learning")
    
    # Plot 2: Parameter evolution
    print("\n2. Parameter convergence...")
    plot_parameter_evolution(result['history'], true_params, "Parameter Evolution")
    
    # Plot 3: Final comparison
    print("\n3. Learned vs Target...")
    compare_learned_vs_target(learned_params, target_data, current, time_config)
    
    # Try with momentum
    print("\n" + "="*60)
    print("Now trying with MOMENTUM for comparison...")
    print("="*60)
    
    result_momentum = gradient_descent_with_momentum(
        initial_params, target_data, current, time_config,
        learning_rate=0.5,
        momentum=0.9,
        max_iterations=50,
        tolerance=0.01,
        params_to_optimize=['tau', 'v_threshold'],
        verbose=True
    )
    
    # Compare both methods
    print("\n" + "="*60)
    print("COMPARISON: Regular vs Momentum")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(result['history']['loss'], 'b-', linewidth=2, label='Regular GD')
    ax1.plot(result_momentum['history']['loss'], 'r-', linewidth=2, label='With Momentum')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves Comparison')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.text(0.5, 0.8, f"Regular GD:", ha='center', fontsize=12, weight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.7, f"Final loss: {result['final_loss']:.6f}", ha='center', fontsize=10, transform=ax2.transAxes)
    ax2.text(0.5, 0.6, f"Iterations: {result['iterations']}", ha='center', fontsize=10, transform=ax2.transAxes)
    
    ax2.text(0.5, 0.4, f"With Momentum:", ha='center', fontsize=12, weight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.3, f"Final loss: {result_momentum['final_loss']:.6f}", ha='center', fontsize=10, transform=ax2.transAxes)
    ax2.text(0.5, 0.2, f"Iterations: {result_momentum['iterations']}", ha='center', fontsize=10, transform=ax2.transAxes)
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)
    print("\nWhat we achieved:")
    print("   • Built a differentiable neuron simulator")
    print("   • Defined loss functions to measure error")
    print("   • Computed gradients numerically")
    print("   • Implemented gradient descent learning")
    print("   • Successfully recovered hidden parameters!")
    print("\nThis is the essence of differentiable programming:")
    print("   Build differentiable systems → compute gradients → learn!")

if __name__ == "__main__":
    main()