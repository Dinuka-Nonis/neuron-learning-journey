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


def get_adaptive_learning_rate(param_name, iteration):
    """
    Get adaptive learning rate for each parameter.
    
    FIXED: Parameter-specific learning rates that decay over time.
    
    Args:
        param_name (str): Parameter name
        iteration (int): Current iteration
        
    Returns:
        float: Learning rate for this parameter at this iteration
    """
    # Base learning rates (empirically tuned)
    base_rates = {
        'tau': 0.5,           # tau is ~20, can take bigger steps
        'v_rest': 0.2,        # voltage ~-70mV
        'v_threshold': 0.05,  # threshold is sensitive!
        'v_reset': 0.2        # reset ~-80mV
    }
    
    base_lr = base_rates.get(param_name, 0.1)
    
    # Decay schedule: lr = base_lr / (1 + decay * iteration)
    decay = 0.001
    lr = base_lr / (1 + decay * iteration)
    
    return lr


def gradient_descent(initial_params, target_data, current, time_config,
                     learning_rate=0.1, max_iterations=100, tolerance=0.01,
                     params_to_optimize=None, verbose=True, adaptive_lr=True):
    """
    Optimize neuron parameters using gradient descent.
    
    FIXED VERSION with:
    - Proper learning rates
    - Adaptive learning rate option
    - Better convergence detection
    - Loss tracking for early stopping
    
    Args:
        initial_params (dict): Starting parameter values
        target_data (dict): Target data to match
        current (np.ndarray): Input current
        time_config (dict): Time configuration
        learning_rate (float): Base step size (default: 0.1)
        max_iterations (int): Maximum iterations (default: 100)
        tolerance (float): Stop if loss < tolerance (default: 0.01)
        params_to_optimize (list): Which parameters to optimize
        verbose (bool): Print progress
        adaptive_lr (bool): Use adaptive learning rates
    """
    
    params = initial_params.copy()
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_rest', 'v_threshold', 'v_reset']
    
    # History tracking
    history = {
        'loss': [],
        'params': {name: [] for name in params_to_optimize},
        'gradients': {name: [] for name in params_to_optimize}
    }
    
    if verbose:
        print("\n" + "="*70)
        print("GRADIENT DESCENT OPTIMIZATION (FIXED)")
        print("="*70)
        print(f"\nSettings:")
        print(f"  Base learning rate: {learning_rate}")
        print(f"  Adaptive LR: {adaptive_lr}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Optimizing: {params_to_optimize}")
        print("\nStarting optimization...\n")
    
    # For early stopping
    best_loss = float('inf')
    patience = 20
    no_improvement_count = 0
    
    # Main learning loop
    for iteration in range(max_iterations):
        # Step 1: Simulate with current parameters
        v_initial = params['v_rest']
        voltage, spikes = simulate_neuron_euler(
            params, time_config, current, v_initial
        )
        simulated = {
            'voltage': voltage,
            'spike_times': spikes,
            'time_config': time_config
        }
        
        # Step 2: Compute loss
        loss_result = compute_combined_loss(simulated, target_data, params)
        loss = loss_result['total']
        
        # Record history
        history['loss'].append(loss)
        for param_name in params_to_optimize:
            history['params'][param_name].append(params[param_name])
        
        # Print progress
        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print(f"Iteration {iteration:3d}: Loss = {loss:.6f}")
            if iteration % 20 == 0 and iteration > 0:
                for param_name in params_to_optimize:
                    print(f"  {param_name:12s} = {params[param_name]:8.3f}")
        
        # Check convergence
        if loss < tolerance:
            if verbose:
                print(f"\n✓ Converged! Loss < {tolerance} at iteration {iteration}")
            break
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            if verbose:
                print(f"\n⚠️  Early stopping: No improvement for {patience} iterations")
            break
        
        # Step 3: Compute gradients (using fixed version from Layer 4)
        gradient_result = compute_all_gradients_finite_diff(
            params, target_data, current, time_config,
            h=None,  # Use optimal step sizes
            params_to_optimize=params_to_optimize
        )
        gradients = gradient_result['gradients']
        
        # Record gradients
        for param_name in params_to_optimize:
            history['gradients'][param_name].append(gradients[param_name])
        
        # Step 4: Update parameters with adaptive learning rates
        for param_name in params_to_optimize:
            gradient = gradients[param_name]
            
            if adaptive_lr:
                lr = get_adaptive_learning_rate(param_name, iteration)
            else:
                lr = learning_rate
            
            # Gradient descent update: param = param - lr * gradient
            params[param_name] -= lr * gradient
        
        # Safety check: ensure parameters stay in reasonable ranges
        params['tau'] = max(1.0, min(100.0, params['tau']))
        params['v_rest'] = max(-100.0, min(-50.0, params['v_rest']))
        params['v_threshold'] = max(-70.0, min(-40.0, params['v_threshold']))
        params['v_reset'] = max(-100.0, min(-60.0, params['v_reset']))
    
    # Final results
    converged = loss < tolerance
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"\nFinal loss: {loss:.6f}")
        print(f"Iterations: {iteration + 1}")
        print(f"Converged: {converged}")
        print(f"Best loss achieved: {best_loss:.6f}")
        print("\nFinal parameters:")
        for param_name in params_to_optimize:
            print(f"  {param_name:12s} = {params[param_name]:8.3f}")
    
    result = {
        'final_params': params,
        'history': history,
        'converged': converged,
        'iterations': iteration + 1,
        'final_loss': loss,
        'best_loss': best_loss
    }
    
    return result


def gradient_descent_with_momentum(initial_params, target_data, current, time_config,
                                   learning_rate=0.01, momentum=0.9, max_iterations=200,
                                   tolerance=0.01, params_to_optimize=None, verbose=True):
    """
    Gradient descent with momentum (accelerated optimization).
    
    FIXED VERSION with proper learning rates and momentum.
    
    Args:
        initial_params (dict): Starting parameters
        target_data (dict): Target data
        current (np.ndarray): Input current
        time_config (dict): Time configuration
        learning_rate (float): Step size (default: 0.01)
        momentum (float): Momentum coefficient 0-1 (default: 0.9)
        max_iterations (int): Maximum iterations
        tolerance (float): Convergence threshold
        params_to_optimize (list): Parameters to optimize
        verbose (bool): Print progress
    """
    
    params = initial_params.copy()
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_rest', 'v_threshold', 'v_reset']
    
    # Initialize velocity for momentum
    velocity = {name: 0.0 for name in params_to_optimize}
    
    history = {
        'loss': [],
        'params': {name: [] for name in params_to_optimize}
    }
    
    if verbose:
        print("\n" + "="*70)
        print("GRADIENT DESCENT WITH MOMENTUM (FIXED)")
        print("="*70)
        print(f"\nSettings:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Momentum: {momentum}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Optimizing: {params_to_optimize}")
        print("\nStarting optimization...\n")
    
    best_loss = float('inf')
    patience = 30
    no_improvement_count = 0
    
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
        
        # Compute loss
        loss_result = compute_combined_loss(simulated, target_data, params)
        loss = loss_result['total']
        
        # Record
        history['loss'].append(loss)
        for param_name in params_to_optimize:
            history['params'][param_name].append(params[param_name])
        
        # Progress
        if verbose and (iteration % 20 == 0 or iteration == max_iterations - 1):
            print(f"Iteration {iteration:4d}: Loss = {loss:.6f}")
        
        # Convergence check
        if loss < tolerance:
            if verbose:
                print(f"\n✓ Converged at iteration {iteration}")
            break
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            if verbose:
                print(f"\n⚠️  Early stopping at iteration {iteration}")
            break
        
        # Compute gradients (using fixed version)
        gradient_result = compute_all_gradients_finite_diff(
            params, target_data, current, time_config,
            h=None,
            params_to_optimize=params_to_optimize
        )
        gradients = gradient_result['gradients']
        
        # Update with momentum
        # v = momentum * v - lr * gradient
        # param = param + v
        for param_name in params_to_optimize:
            velocity[param_name] = (momentum * velocity[param_name] - 
                                   learning_rate * gradients[param_name])
            params[param_name] += velocity[param_name]
        
        # Safety constraints
        params['tau'] = max(1.0, min(100.0, params['tau']))
        params['v_rest'] = max(-100.0, min(-50.0, params['v_rest']))
        params['v_threshold'] = max(-70.0, min(-40.0, params['v_threshold']))
        params['v_reset'] = max(-100.0, min(-60.0, params['v_reset']))
    
    converged = loss < tolerance
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Final loss: {loss:.6f} after {iteration + 1} iterations")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Converged: {converged}")
    
    result = {
        'final_params': params,
        'history': history,
        'converged': converged,
        'iterations': iteration + 1,
        'final_loss': loss,
        'best_loss': best_loss
    }
    
    return result


def plot_learning_curve(history, title="Learning Curve"):
    """Plot loss over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def plot_parameter_evolution(history, true_params=None, title="Parameter Evolution"):
    """Plot how parameters change over iterations."""
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
    """Compare learned parameters against target."""
    v_initial = learned_params['v_rest']
    voltage_learned, spikes_learned = simulate_neuron_euler(
        learned_params, time_config, current, v_initial
    )
    
    voltage_target = target_data['voltage']
    spikes_target = target_data['spike_times']
    time = target_data['time']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
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
    
    ax2.plot(time, current, 'orange', linewidth=2)
    ax2.fill_between(time, 0, current, alpha=0.3, color='orange')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Input Current', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print("\n" + "="*60)
    print("LEARNED VS TARGET COMPARISON")
    print("="*60)
    print(f"\nSpikes:")
    print(f"  Target:  {len(spikes_target)} spikes")
    print(f"  Learned: {len(spikes_learned)} spikes")
    print(f"  Match:   {len(spikes_target) == len(spikes_learned)}")
    
    mse = np.mean((voltage_learned - voltage_target)**2)
    print(f"\nVoltage MSE: {mse:.6f} mV²")
    
    true_params = target_data['params']
    print(f"\nParameters:")
    print(f"{'Parameter':<15} {'True':>10} {'Learned':>10} {'Error':>10} {'Error %':>10}")
    print("-"*60)
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        true_val = true_params[param_name]
        learned_val = learned_params[param_name]
        error = abs(learned_val - true_val)
        error_pct = (error / abs(true_val)) * 100 if true_val != 0 else 0
        print(f"{param_name:<15} {true_val:10.2f} {learned_val:10.2f} {error:10.4f} {error_pct:10.2f}%")


def main():
    """
    Demonstrate FIXED learning with proper hyperparameters.
    """
    print("="*70)
    print("LAYER 5: FIXED PARAMETER LEARNING")
    print("="*70)
    print("\nKey Fixes:")
    print("1. ✓ Proper learning rates (0.05-0.5 instead of 1e-8)")
    print("2. ✓ Adaptive learning rates per parameter")
    print("3. ✓ Early stopping for efficiency")
    print("4. ✓ Parameter constraints to avoid divergence")
    
    # Setup
    print("\n📊 Setting up simulation...")
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    current = create_constant_inputs(time, amplitude=20.0)
    
    # Generate target
    print("📊 Generating target data...")
    target_data = generate_target_data(current, time_config, noise_level=0.0)
    
    true_params = target_data['params']
    print(f"\nTrue parameters (hidden):")
    for param in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param}: {true_params[param]:.2f}")
    
    initial_params = get_default_parameters()
    print(f"\nInitial guess:")
    for param in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param}: {initial_params[param]:.2f}")
    
    print(f"\nTarget has {len(target_data['spike_times'])} spikes")
    
    # Method 1: Standard gradient descent with adaptive LR
    print("\n" + "="*70)
    print("METHOD 1: Standard Gradient Descent (Adaptive LR)")
    print("="*70)
    
    result1 = gradient_descent(
        initial_params, target_data, current, time_config,
        learning_rate=0.1,
        max_iterations=100,
        tolerance=0.01,
        params_to_optimize=['tau', 'v_threshold'],
        verbose=True,
        adaptive_lr=True
    )
    
    print("\n📊 Plotting results...")
    plot_learning_curve(result1['history'], "Method 1: Standard GD")
    plot_parameter_evolution(result1['history'], true_params, "Method 1: Parameter Evolution")
    compare_learned_vs_target(result1['final_params'], target_data, current, time_config)
    
    # Method 2: Momentum
    print("\n" + "="*70)
    print("METHOD 2: Gradient Descent with Momentum")
    print("="*70)
    
    result2 = gradient_descent_with_momentum(
        initial_params, target_data, current, time_config,
        learning_rate=0.05,
        momentum=0.9,
        max_iterations=150,
        tolerance=0.01,
        params_to_optimize=['tau', 'v_threshold'],
        verbose=True
    )
    
    print("\n📊 Plotting results...")
    plot_learning_curve(result2['history'], "Method 2: Momentum")
    plot_parameter_evolution(result2['history'], true_params, "Method 2: Parameter Evolution")
    compare_learned_vs_target(result2['final_params'], target_data, current, time_config)
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nMethod 1 (Standard GD):")
    print(f"  Final loss: {result1['final_loss']:.6f}")
    print(f"  Iterations: {result1['iterations']}")
    print(f"  Converged: {result1['converged']}")
    
    print(f"\nMethod 2 (Momentum):")
    print(f"  Final loss: {result2['final_loss']:.6f}")
    print(f"  Iterations: {result2['iterations']}")
    print(f"  Converged: {result2['converged']}")
    
    print("\n✅ LEARNING COMPLETE WITH FIXED HYPERPARAMETERS!")


if __name__ == "__main__":
    main()