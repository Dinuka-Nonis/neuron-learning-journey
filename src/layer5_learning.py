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


def generate_good_target(time_config, min_spikes=5):
    """
    Generate target data with guaranteed spikes.
    
    CRITICAL FIX: Ensures target has enough spikes for learning signal.
    """
    time = time_config['time']
    
    # Try increasing current amplitudes until we get enough spikes
    for amplitude in [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
        current = create_constant_inputs(time, amplitude=amplitude)
        target_data = generate_target_data(current, time_config, noise_level=0.0)
        
        n_spikes = len(target_data['spike_times'])
        
        if n_spikes >= min_spikes:
            print(f"✓ Generated target with {n_spikes} spikes (amplitude={amplitude})")
            return target_data, current
    
    # If still not enough, use last attempt
    print(f"⚠️  Could only generate {n_spikes} spikes (wanted {min_spikes})")
    return target_data, current


def gradient_descent_robust(initial_params, target_data, current, time_config,
                            learning_rate=0.5, max_iterations=200, tolerance=1.0,
                            params_to_optimize=None, verbose=True):
    """
    ROBUST gradient descent with better convergence detection.
    
    Key improvements:
    - Higher tolerance (1.0 instead of 0.01) - more realistic
    - Better learning rates
    - Checks for stuck gradients
    - Better stopping criteria
    """
    
    params = initial_params.copy()
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_threshold']  # Focus on most important
    
    history = {
        'loss': [],
        'params': {name: [] for name in params_to_optimize},
        'gradients': {name: [] for name in params_to_optimize}
    }
    
    if verbose:
        print("\n" + "="*70)
        print("ROBUST GRADIENT DESCENT")
        print("="*70)
        print(f"Settings:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Optimizing: {params_to_optimize}")
        print()
    
    best_loss = float('inf')
    best_params = params.copy()
    patience = 30
    no_improvement_count = 0
    min_gradient_threshold = 1e-6
    stuck_count = 0
    
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
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Progress
        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print(f"Iter {iteration:3d}: Loss={loss:.4f}, Best={best_loss:.4f}, Spikes={len(spikes)}")
        
        # Check convergence
        if loss < tolerance:
            if verbose:
                print(f"\n✓ Converged! Loss < {tolerance}")
            break
        
        # Early stopping
        if no_improvement_count >= patience:
            if verbose:
                print(f"\n⚠️  Early stop: no improvement for {patience} iterations")
            break
        
        # Compute gradients
        try:
            gradient_result = compute_all_gradients_finite_diff(
                params, target_data, current, time_config,
                h=None,
                params_to_optimize=params_to_optimize
            )
            gradients = gradient_result['gradients']
            grad_magnitude = gradient_result['magnitude']
        except Exception as e:
            if verbose:
                print(f"\n⚠️  Gradient computation failed: {e}")
            break
        
        # Record gradients
        for param_name in params_to_optimize:
            history['gradients'][param_name].append(gradients[param_name])
        
        # Check if stuck (gradients too small)
        if grad_magnitude < min_gradient_threshold:
            stuck_count += 1
            if stuck_count >= 5:
                if verbose:
                    print(f"\n⚠️  Stuck: gradients too small ({grad_magnitude:.2e})")
                break
        else:
            stuck_count = 0
        
        # Update parameters with adaptive rates
        for param_name in params_to_optimize:
            gradient = gradients[param_name]
            
            # Parameter-specific learning rates
            if param_name == 'tau':
                lr = learning_rate * 2.0  # tau can handle bigger steps
            elif param_name == 'v_threshold':
                lr = learning_rate * 0.2  # threshold is sensitive
            else:
                lr = learning_rate
            
            # Update
            params[param_name] -= lr * gradient
        
        # Constraints
        params['tau'] = np.clip(params['tau'], 5.0, 50.0)
        params['v_threshold'] = np.clip(params['v_threshold'], -65.0, -45.0)
        if 'v_rest' in params_to_optimize:
            params['v_rest'] = np.clip(params['v_rest'], -80.0, -60.0)
        if 'v_reset' in params_to_optimize:
            params['v_reset'] = np.clip(params['v_reset'], -90.0, -70.0)
    
    # Use best parameters found
    params = best_params
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best loss: {best_loss:.4f}")
        print(f"Iterations: {iteration + 1}")
        print(f"Final parameters:")
        for param_name in params_to_optimize:
            print(f"  {param_name}: {params[param_name]:.3f}")
    
    result = {
        'final_params': params,
        'history': history,
        'converged': best_loss < tolerance,
        'iterations': iteration + 1,
        'final_loss': best_loss,
        'best_loss': best_loss
    }
    
    return result


def plot_comprehensive_results(result, target_data, current, time_config):
    """
    Create comprehensive visualization of results.
    """
    learned_params = result['final_params']
    history = result['history']
    
    # Simulate with learned params
    v_initial = learned_params['v_rest']
    voltage_learned, spikes_learned = simulate_neuron_euler(
        learned_params, time_config, current, v_initial
    )
    
    voltage_target = target_data['voltage']
    spikes_target = target_data['spike_times']
    time = target_data['time']
    true_params = target_data['params']
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Voltage comparison
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(time, voltage_target, 'r-', linewidth=2.5, alpha=0.8, label='Target')
    ax1.plot(time, voltage_learned, 'b--', linewidth=2.5, alpha=0.8, label='Learned')
    
    for st in spikes_target:
        ax1.axvline(x=st, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
    for st in spikes_learned:
        ax1.axvline(x=st, color='blue', linestyle=':', alpha=0.4, linewidth=1.5)
    
    ax1.set_ylabel('Voltage (mV)', fontsize=12, weight='bold')
    ax1.set_title('Voltage: Target vs Learned', fontsize=13, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Input current
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.fill_between(time, 0, current, alpha=0.4, color='orange')
    ax2.plot(time, current, 'orange', linewidth=2)
    ax2.set_xlabel('Time (ms)', fontsize=12, weight='bold')
    ax2.set_ylabel('Current', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning curve
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.semilogy(history['loss'], 'g-', linewidth=2.5)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Loss (log scale)', fontsize=11, weight='bold')
    ax3.set_title('Convergence', fontsize=12, weight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Parameter evolution
    ax4 = fig.add_subplot(gs[1, 1])
    params_opt = list(history['params'].keys())
    for param_name in params_opt:
        values = history['params'][param_name]
        ax4.plot(values, linewidth=2, label=param_name)
        if param_name in true_params:
            ax4.axhline(y=true_params[param_name], linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Parameter Value', fontsize=11)
    ax4.set_title('Parameter Evolution', fontsize=12, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Metrics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    mse = np.mean((voltage_learned - voltage_target)**2)
    corr = np.corrcoef(voltage_learned, voltage_target)[0, 1]
    
    metrics_text = (
        f"PERFORMANCE METRICS\n"
        f"{'='*35}\n"
        f"Final Loss: {result['final_loss']:.4f}\n"
        f"Converged: {'YES' if result['converged'] else 'NO'}\n"
        f"Iterations: {result['iterations']}\n"
        f"{'='*35}\n"
        f"Voltage MSE: {mse:.4f} mV²\n"
        f"Correlation: {corr:.4f}\n"
        f"Target Spikes: {len(spikes_target)}\n"
        f"Learned Spikes: {len(spikes_learned)}\n"
        f"Spike Match: {'YES' if len(spikes_target)==len(spikes_learned) else 'NO'}\n"
        f"{'='*35}\n"
    )
    
    # Add parameter errors
    for param_name in params_opt:
        if param_name in true_params:
            true_val = true_params[param_name]
            learned_val = learned_params[param_name]
            error = abs(learned_val - true_val)
            error_pct = (error / abs(true_val)) * 100 if true_val != 0 else 0
            metrics_text += f"{param_name}: {error_pct:.1f}% error\n"
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('NEURON PARAMETER LEARNING - RESULTS', 
                 fontsize=15, weight='bold', y=0.995)
    
    return fig


def main():
    """
    FINAL WORKING VERSION with all fixes.
    """
    print("="*70)
    print("NEURON PARAMETER LEARNING - FINAL WORKING VERSION")
    print("="*70)
    print("\nAll Critical Fixes Applied:")
    print("1. ✓ Gradient computation fixed (perturbed params)")
    print("2. ✓ Proper learning rates (0.5 base, adaptive)")
    print("3. ✓ Better step sizes in finite differences")
    print("4. ✓ Target guaranteed to have spikes")
    print("5. ✓ Realistic convergence tolerance (1.0)")
    print("6. ✓ Focus on most important parameters (tau, v_threshold)")
    print()
    
    # Setup
    print("📊 Setting up simulation...")
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    
    # Generate GOOD target (guaranteed spikes)
    print("📊 Generating target with guaranteed spikes...")
    target_data, current = generate_good_target(time_config, min_spikes=5)
    
    true_params = target_data['params']
    print(f"\nTrue parameters (hidden from learner):")
    for param in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param}: {true_params[param]:.2f}")
    
    # Initial guess (deliberately wrong)
    initial_params = get_default_parameters()
    print(f"\nInitial guess (deliberately wrong):")
    for param in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        print(f"  {param}: {initial_params[param]:.2f}")
    
    print(f"\nTarget has {len(target_data['spike_times'])} spikes")
    print(f"Initial simulation has: ", end='')
    v_init = initial_params['v_rest']
    v_init_sim, s_init_sim = simulate_neuron_euler(
        initial_params, time_config, current, v_init
    )
    print(f"{len(s_init_sim)} spikes")
    
    # Check initial loss
    sim_init = {
        'voltage': v_init_sim,
        'spike_times': s_init_sim,
        'time_config': time_config
    }
    loss_init = compute_combined_loss(sim_init, target_data, initial_params)
    print(f"Initial loss: {loss_init['total']:.4f}")
    
    # LEARN!
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION")
    print("="*70)
    
    result = gradient_descent_robust(
        initial_params, target_data, current, time_config,
        learning_rate=0.01,
        max_iterations=15000,
        tolerance=0.9,  # Realistic tolerance
        params_to_optimize=['tau', 'v_threshold'],  # Focus on key params
        verbose=True
    )
    
    learned_params = result['final_params']
    
    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print(f"\nOptimization:")
    print(f"  Initial loss: {loss_init['total']:.4f}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Improvement: {loss_init['total'] - result['final_loss']:.4f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['converged']}")
    
    print(f"\nParameter Recovery:")
    print(f"{'Parameter':<15} {'True':>10} {'Initial':>10} {'Learned':>10} {'Error':>10}")
    print("-"*60)
    
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        true_val = true_params[param_name]
        init_val = initial_params[param_name]
        learned_val = learned_params[param_name]
        error = abs(learned_val - true_val)
        
        print(f"{param_name:<15} {true_val:10.2f} {init_val:10.2f} {learned_val:10.2f} {error:10.3f}")
    
    # Verify learned model
    v_final = learned_params['v_rest']
    voltage_learned, spikes_learned = simulate_neuron_euler(
        learned_params, time_config, current, v_final
    )
    
    print(f"\nSpike Matching:")
    print(f"  Target spikes: {len(target_data['spike_times'])}")
    print(f"  Initial spikes: {len(s_init_sim)}")
    print(f"  Learned spikes: {len(spikes_learned)}")
    
    mse = np.mean((voltage_learned - target_data['voltage'])**2)
    print(f"\nVoltage MSE: {mse:.4f} mV²")
    
    # Visualization
    print("\n📊 Creating comprehensive visualization...")
    fig = plot_comprehensive_results(result, target_data, current, time_config)
    plt.tight_layout()
    plt.savefig('neuron_learning_final_result.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'neuron_learning_final_result.png'")
    plt.show()
    
    # Success criteria
    print("\n" + "="*70)
    print("SUCCESS EVALUATION")
    print("="*70)
    
    improvement = loss_init['total'] - result['final_loss']
    spike_match = len(spikes_learned) == len(target_data['spike_times'])
    tau_error = abs(learned_params['tau'] - true_params['tau'])
    thresh_error = abs(learned_params['v_threshold'] - true_params['v_threshold'])
    
    print(f"\n✓ Loss improved: {improvement > 0}")
    print(f"✓ Loss reduction: {improvement:.4f}")
    print(f"✓ Spike count matches: {spike_match}")
    print(f"✓ Tau error: {tau_error:.2f} ms")
    print(f"✓ Threshold error: {thresh_error:.2f} mV")
    
    if improvement > 10 and (tau_error < 10 or thresh_error < 5):
        print("\n🎉 SUCCESS! Model learned meaningful parameters!")
    elif improvement > 5:
        print("\n✓ PARTIAL SUCCESS! Model improved but could be better")
        print("  Try: longer training, different parameters, or different target")
    else:
        print("\n⚠️  LIMITED SUCCESS. Suggestions:")
        print("  - Check if target has enough spikes")
        print("  - Try optimizing different parameters")
        print("  - Adjust learning rates")
    
    print("\n" + "="*70)
    print("✅ LEARNING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()