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


def gradient_descent_with_momentum(initial_params, target_data, current, time_config,
                                   learning_rate=0.001, momentum=0.9, max_iterations=1000, 
                                   tolerance=0.01, params_to_optimize=None, verbose=True):
    """
    Gradient descent with momentum (accelerated optimization)
    
    Uses the FIXED gradient computation from Layer 4
    """

    params = initial_params.copy()
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_rest', 'v_threshold', 'v_reset']
    
    velocity = {name: 0.0 for name in params_to_optimize}
    
    history = {
        'loss': [],
        'params': {name: [] for name in params_to_optimize}
    }
    
    if verbose:
        print("\n" + "="*70)
        print("GRADIENT DESCENT WITH MOMENTUM")
        print("="*70)
        print(f"\nSettings:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Momentum: {momentum}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Optimizing: {params_to_optimize}")
        print("\nStarting optimization...\n")
    
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
        
        # Progress
        if verbose and (iteration % 50 == 0 or iteration == max_iterations - 1):
            print(f"Iteration {iteration:4d}: Loss = {loss:.6f}")
        
        # Check convergence
        if loss < tolerance:
            if verbose:
                print(f"\n✅ Converged at iteration {iteration}")
            break
        
        # Step 3: Compute gradients (FIXED VERSION)
        gradient_result = compute_all_gradients_finite_diff(
            params, target_data, current, time_config,
            h=None,  # Auto-calculate optimal step size
            params_to_optimize=params_to_optimize
        )
        gradients = gradient_result['gradients']
        
        # Step 4: Update with momentum
        for param_name in params_to_optimize:
            velocity[param_name] = (momentum * velocity[param_name] - 
                                   learning_rate * gradients[param_name])
            params[param_name] += velocity[param_name]
    
    converged = loss < tolerance
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Final loss: {loss:.6f} after {iteration + 1} iterations")
        print(f"Converged: {converged}")
    
    result = {
        'final_params': params,
        'history': history,
        'converged': converged,
        'iterations': iteration + 1,
        'final_loss': loss
    }
    
    return result


def create_performance_summary(result, target_data, learned_params, current, time_config):
    """
    Create a comprehensive professional summary figure showing model performance.
    """
    v_initial = learned_params['v_rest']
    voltage_learned, spikes_learned = simulate_neuron_euler(
        learned_params, time_config, current, v_initial
    )
    
    voltage_target = target_data['voltage']
    spikes_target = target_data['spike_times']
    time = target_data['time']
    true_params = target_data['params']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Main comparison: Voltage traces
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(time, voltage_target, 'r-', linewidth=2.5, alpha=0.8, label='Target (True Model)')
    ax1.plot(time, voltage_learned, 'b--', linewidth=2.5, alpha=0.8, label='Learned Model')
    
    for spike_time in spikes_target:
        ax1.axvline(x=spike_time, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
    for spike_time in spikes_learned:
        ax1.axvline(x=spike_time, color='blue', linestyle=':', alpha=0.4, linewidth=1.5)
    
    ax1.set_ylabel('Voltage (mV)', fontsize=12, weight='bold')
    ax1.set_title('Neuron Voltage: Target vs Learned Model', fontsize=13, weight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time[0], time[-1]])
    
    # 2. Input current
    ax2 = fig.add_subplot(gs[2, 0:2])
    ax2.fill_between(time, 0, current, alpha=0.4, color='orange')
    ax2.plot(time, current, 'orange', linewidth=2)
    ax2.set_xlabel('Time (ms)', fontsize=12, weight='bold')
    ax2.set_ylabel('Current (uA)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time[0], time[-1]])
    
    # 3. Learning curve (loss)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(result['history']['loss'], 'g-', linewidth=2.5, marker='o', markersize=4)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11, weight='bold')
    ax3.set_title('Convergence', fontsize=12, weight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Parameter errors
    ax4 = fig.add_subplot(gs[1, 2])
    params_optimized = list(result['history']['params'].keys())
    param_errors = {}
    
    for param_name in params_optimized:
        true_val = true_params[param_name]
        learned_val = learned_params[param_name]
        error_pct = (abs(learned_val - true_val) / abs(true_val)) * 100 if true_val != 0 else 0
        param_errors[param_name] = error_pct
    
    colors = ['#2ecc71' if e < 5 else '#f39c12' if e < 15 else '#e74c3c' for e in param_errors.values()]
    bars = ax4.barh(list(param_errors.keys()), list(param_errors.values()), color=colors, alpha=0.7)
    ax4.set_xlabel('Error (%)', fontsize=11, weight='bold')
    ax4.set_title('Parameter Accuracy', fontsize=12, weight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    for i, (param, error) in enumerate(param_errors.items()):
        ax4.text(error + 0.5, i, f'{error:.1f}%', va='center', fontsize=10, weight='bold')
    
    # 5. Metrics box
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    mse = np.mean((voltage_learned - voltage_target)**2)
    spike_accuracy = (len(spikes_target) == len(spikes_learned))
    corr = np.corrcoef(voltage_learned, voltage_target)[0, 1]
    
    metrics_text = (
        f"PERFORMANCE METRICS\n"
        f"{'='*30}\n"
        f"Voltage MSE: {mse:.4f} mV^2\n"
        f"Correlation: {corr:.4f}\n"
        f"Target Spikes: {len(spikes_target)}\n"
        f"Learned Spikes: {len(spikes_learned)}\n"
        f"Spike Match: {'YES' if spike_accuracy else 'NO'}\n"
        f"{'='*30}\n"
        f"Converged: {'YES' if result['converged'] else 'NO'}\n"
        f"Iterations: {result['iterations']}\n"
        f"Final Loss: {result['final_loss']:.4f}"
    )
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('NEURON PARAMETER LEARNING - PERFORMANCE SUMMARY', 
                 fontsize=15, weight='bold', y=0.995)
    
    return fig


def main():
    """
    Complete learning demonstration with FIXED gradients - BEST RESULT ONLY
    """
    print("="*70)
    print("NEURON PARAMETER LEARNING - FIXED GRADIENT COMPUTATION")
    print("="*70)
    print("\nUsing improved gradient calculation from Layer 4")
    print("Focus: Get the BEST possible model fit\n")
    
    # Setup
    print("Setting up simulation...")
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    current = create_constant_inputs(time, amplitude=35.0)
    
    # Generate target
    print("Generating target data...")
    target_data = generate_target_data(current, time_config, noise_level=0.0)
    
    target_spikes = target_data['spike_times']
    if len(target_spikes) == 0:
        print("Retrying with higher current...")
        current = create_constant_inputs(time, amplitude=50.0)
        target_data = generate_target_data(current, time_config, noise_level=0.0)
        target_spikes = target_data['spike_times']
    
    true_params = target_data['params']
    print(f"\nTrue Parameters (Target - Hidden):")
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param_name:12s} = {true_params[param_name]:8.4f}")
    
    initial_params = get_default_parameters()
    print(f"\nInitial Parameters (Bad Guess):")
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param_name:12s} = {initial_params[param_name]:8.4f}")
    
    print(f"\nTarget has {len(target_spikes)} spikes")
    
    # LEARNING - Optimize tau and v_rest for smooth voltage matching
    print("\n" + "="*70)
    print("LEARNING: Optimizing tau and v_rest")
    print("="*70)
    
    result = gradient_descent_with_momentum(
        initial_params, target_data, current, time_config,
        learning_rate=0.00000001,  # INCREASED - let it learn faster
        momentum=0.8,
        max_iterations=100000,  # More iterations
        tolerance=0.1,
        params_to_optimize=['tau', 'v_rest', 'v_threshold'],  # Only these two
        verbose=True
    )
    
    learned_params = result['final_params']
    
    # RESULTS
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nOptimization Summary:")
    print(f"  Final Loss: {result['final_loss']:.6f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['converged']}")
    
    print(f"\nLearned Parameters:")
    param_errors = {}
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        true_val = true_params[param_name]
        learned_val = learned_params[param_name]
        error = abs(learned_val - true_val)
        error_pct = (error / abs(true_val)) * 100 if true_val != 0 else 0
        param_errors[param_name] = error_pct
        
        status = "Excellent" if error_pct < 5 else "Good" if error_pct < 15 else "Fair"
        print(f"  {param_name:12s} = {learned_val:8.4f} (error: {error_pct:6.2f}%) [{status}]")
    
    # VISUALIZATION
    print("\n" + "="*70)
    print("GENERATING PROFESSIONAL VISUALIZATION")
    print("="*70)
    
    fig = create_performance_summary(result, target_data, learned_params, current, time_config)
    plt.tight_layout()
    plt.savefig('best_result_summary.png', dpi=300, bbox_inches='tight')
    print("\nSaved as 'best_result_summary.png' (high resolution)")
    plt.show()
    
    # DETAILED ANALYSIS
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    v_initial = learned_params['v_rest']
    voltage_learned, spikes_learned = simulate_neuron_euler(
        learned_params, time_config, current, v_initial
    )
    
    voltage_target = target_data['voltage']
    spikes_target = target_data['spike_times']
    
    mse = np.mean((voltage_learned - voltage_target)**2)
    max_error = np.max(np.abs(voltage_learned - voltage_target))
    corr = np.corrcoef(voltage_learned, voltage_target)[0, 1]
    
    print(f"\nVoltage Reconstruction Quality:")
    print(f"  MSE: {mse:.6f} mV^2")
    print(f"  Max Error: {max_error:.4f} mV")
    print(f"  Correlation: {corr:.6f}")
    
    print(f"\nSpike Matching:")
    print(f"  Target Spikes: {len(spikes_target)}")
    print(f"  Learned Spikes: {len(spikes_learned)}")
    print(f"  Match: {'YES' if len(spikes_target) == len(spikes_learned) else 'NO'}")
    
    print(f"\nParameter Recovery Summary:")
    avg_error = np.mean(list(param_errors.values()))
    print(f"  Average Error: {avg_error:.2f}%")
    excellent = sum(1 for e in param_errors.values() if e < 5)
    print(f"  Parameters within 5%: {excellent}/4")
    
    print("\n" + "="*70)
    print("SUCCESS! Model trained with fixed gradients!")
    print("="*70)


if __name__ == "__main__":
    main()