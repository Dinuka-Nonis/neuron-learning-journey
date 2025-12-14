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

def compute_gradient_finite_diff(param_name, params, target_data, current, time_config, h=None):
    """
    Compute gradient of loss w.r.t. ONE parameter using finite differences
    
    FIXED VERSION - Proper numerical differentiation
    """
    
    # Auto-determine step size if not provided (5% of parameter value)
    if h is None:
        h = abs(params[param_name]) * 0.05
        if h < 1e-6:  # Prevent tiny steps
            h = 0.1
    
    print(f"\n  Computing ∂loss/∂{param_name}... (h={h:.4f})", end='')
    
    # Step 1: Compute loss at ORIGINAL parameters
    v_initial = params['v_rest']
    voltage_original, spikes_original = simulate_neuron_euler(
        params, time_config, current, v_initial
    )
    simulated_original = {
        'voltage': voltage_original,
        'spike_times': spikes_original,
        'time_config': time_config
    }
    
    loss_result_original = compute_combined_loss(
        simulated_original, target_data, params  # FIXED: pass original params
    )
    loss_original = loss_result_original['total']
    
    # Step 2: Perturb parameter UPWARD by h
    params_perturbed_plus = params.copy()
    original_value = params[param_name]
    params_perturbed_plus[param_name] = original_value + h
    
    # Step 3: Compute loss at PERTURBED+h
    voltage_perturbed_plus, spikes_perturbed_plus = simulate_neuron_euler(
        params_perturbed_plus, time_config, current, v_initial
    )
    
    simulated_perturbed_plus = {
        'voltage': voltage_perturbed_plus,
        'spike_times': spikes_perturbed_plus,
        'time_config': time_config
    }
    
    loss_result_perturbed_plus = compute_combined_loss(
        simulated_perturbed_plus, target_data, params  # FIXED: pass original params
    )
    loss_perturbed_plus = loss_result_perturbed_plus['total']
    
    # Step 4: Perturb parameter DOWNWARD by h (for better accuracy)
    params_perturbed_minus = params.copy()
    params_perturbed_minus[param_name] = original_value - h
    
    voltage_perturbed_minus, spikes_perturbed_minus = simulate_neuron_euler(
        params_perturbed_minus, time_config, current, v_initial
    )
    
    simulated_perturbed_minus = {
        'voltage': voltage_perturbed_minus,
        'spike_times': spikes_perturbed_minus,
        'time_config': time_config
    }
    
    loss_result_perturbed_minus = compute_combined_loss(
        simulated_perturbed_minus, target_data, params  # FIXED: pass original params
    )
    loss_perturbed_minus = loss_result_perturbed_minus['total']
    
    # Step 5: Central finite difference (more accurate than forward difference)
    # gradient ≈ (f(x+h) - f(x-h)) / (2h)
    gradient = (loss_perturbed_plus - loss_perturbed_minus) / (2 * h)
    
    # Sanity check
    loss_change_plus = loss_perturbed_plus - loss_original
    loss_change_minus = loss_perturbed_minus - loss_original
    
    result = {
        'gradient': gradient,
        'loss_original': loss_original,
        'loss_perturbed_plus': loss_perturbed_plus,
        'loss_perturbed_minus': loss_perturbed_minus,
        'param_original': original_value,
        'param_plus': original_value + h,
        'param_minus': original_value - h,
        'h': h,
        'loss_change_plus': loss_change_plus,
        'loss_change_minus': loss_change_minus
    }
    
    # Print details
    if abs(gradient) < 1e-8:
        print(f" ⚠️ WARNING: gradient ≈ 0.0000 (might be stuck or bad h={h:.4f})")
    else:
        direction = "↑" if gradient > 0 else "↓"
        print(f" ✓ gradient = {gradient:+.6f} {direction}")
    
    return result


def compute_all_gradients_finite_diff(params, target_data, current, time_config, 
                                      h=None, params_to_optimize=None):
    """
    Compute gradients for all parameters.
    
    FIXED VERSION - Uses central differences, proper step sizing
    """
    
    if params_to_optimize is None:
        params_to_optimize = ['tau', 'v_rest', 'v_threshold', 'v_reset']
    
    gradients = {}
    details = {}
    
    print("\nComputing gradients with central finite differences...")
    
    for param_name in params_to_optimize:
        result = compute_gradient_finite_diff(
            param_name, params, target_data, current, time_config, h
        )
        
        gradients[param_name] = result['gradient']
        details[param_name] = result
    
    loss_original = details[params_to_optimize[0]]['loss_original']
    
    print(f"\n{'='*60}")
    print("Gradient Summary:")
    print(f"{'='*60}")
    print(f"Current Loss: {loss_original:.6f}\n")
    
    for param_name in params_to_optimize:
        grad = gradients[param_name]
        if grad > 0:
            print(f"  ∂loss/∂{param_name:12s} = {grad:+.8f}  (decrease {param_name})")
        elif grad < 0:
            print(f"  ∂loss/∂{param_name:12s} = {grad:+.8f}  (increase {param_name})")
        else:
            print(f"  ∂loss/∂{param_name:12s} = {grad:+.8f}  (no gradient signal)")
    
    result = {
        'gradients': gradients,
        'loss': loss_original,
        'details': details
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

def normalize_gradient(gradients):
    """
    Normalize gradient vector to unit length.
    
    This gives us the DIRECTION of steepest descent without
    worrying about magnitude.
    
    normalized_grad = grad / ||grad||
    
    Args:
        gradients (dict): Dictionary of gradients {param_name: value}
    
    Returns:
        dict: Normalized gradients (unit vector)
    """
    # Convert to array
    grad_values = np.array(list(gradients.values()))
    
    # Compute magnitude (L2 norm)
    magnitude = np.linalg.norm(grad_values)
    
    if magnitude < 1e-10:
        # Gradient is essentially zero - return as is
        return gradients.copy()
    
    # Normalize
    normalized_values = grad_values / magnitude
    
    # Convert back to dictionary
    normalized_gradients = {}
    for i, param_name in enumerate(gradients.keys()):
        normalized_gradients[param_name] = normalized_values[i]
    
    return normalized_gradients

def print_gradient_info(gradient_result):
    """
    Print gradient information in readable format.
    
    Args:
        gradient_result (dict): Result from compute_all_gradients_finite_diff()
    """
    print("\n" + "="*60)
    print("GRADIENT INFORMATION")
    print("="*60)
    
    gradients = gradient_result['gradients']
    loss = gradient_result['loss']
    
    print(f"\nCurrent Loss: {loss:.4f}")
    
    print("\nGradients (∂loss/∂param):")
    for param_name, grad_value in gradients.items():
        direction = "↗ increase" if grad_value > 0 else "↘ decrease"
        print(f"  ∂loss/∂{param_name:12s} = {grad_value:+10.4f}  ({direction} param to reduce loss)")
    
    # Compute magnitude
    grad_values = np.array(list(gradients.values()))
    magnitude = np.linalg.norm(grad_values)
    print(f"\nGradient Magnitude: ||∇loss|| = {magnitude:.4f}")
    
    # Show normalized gradient
    normalized = normalize_gradient(gradients)
    print("\nNormalized Gradient (direction only):")
    for param_name, norm_value in normalized.items():
        print(f"  {param_name:12s}: {norm_value:+.4f}")
    
    # Suggest update direction
    print("\n" + "-"*60)
    print("To REDUCE loss, update parameters as:")
    for param_name, grad_value in gradients.items():
        sign = "-" if grad_value > 0 else "+"
        print(f"  {param_name:12s} → {sign} (move opposite to gradient)")
    print("="*60 + "\n")

def plot_loss_landscape_1d(param_name, params, target_data, current, 
                           time_config, param_range=None, n_points=20):
    """
    Plot loss landscape for ONE parameter.
    
    This shows how loss changes as we vary a single parameter,
    keeping all others fixed.
    
    Args:
        param_name (str): Parameter to vary
        params (dict): Current parameters
        target_data (dict): Target data
        current (np.ndarray): Input current
        time_config (dict): Time configuration
        param_range (tuple): (min, max) values to plot. If None, uses ±20% of current
        n_points (int): Number of points to evaluate
    """
    # Determine range
    current_value = params[param_name]
    if param_range is None:
        # Default: ±20% of current value
        param_min = current_value * 0.8
        param_max = current_value * 1.2
    else:
        param_min, param_max = param_range
    
    # Sample parameter values
    param_values = np.linspace(param_min, param_max, n_points)
    losses = []
    
    print(f"\nEvaluating loss landscape for {param_name}...")
    
    # Compute loss for each value
    v_initial = params['v_rest']
    for val in param_values:
        # Create modified parameters
        params_modified = params.copy()
        params_modified[param_name] = val
        
        # Simulate
        voltage, spikes = simulate_neuron_euler(
            params_modified, time_config, current, v_initial
        )
        
        # Compute loss
        simulated = {
            'voltage': voltage,
            'spike_times': spikes,
            'time_config': time_config
        }
        loss_result = compute_combined_loss(simulated, target_data, params_modified)
        losses.append(loss_result['total'])
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, losses, 'b-', linewidth=2, label='Loss landscape')
    
    # Mark current value
    current_loss = losses[np.argmin(np.abs(param_values - current_value))]
    plt.plot(current_value, current_loss, 'ro', markersize=12, 
             label=f'Current: {param_name}={current_value:.2f}')
    
    # Mark minimum
    min_idx = np.argmin(losses)
    min_param = param_values[min_idx]
    min_loss = losses[min_idx]
    plt.plot(min_param, min_loss, 'g*', markersize=15, 
             label=f'Minimum: {param_name}={min_param:.2f}')
    
    plt.xlabel(f'{param_name}', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss Landscape: {param_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Current {param_name}: {current_value:.2f}, Loss: {current_loss:.2f}")
    print(f"Optimal {param_name}: {min_param:.2f}, Loss: {min_loss:.2f}")

def plot_loss_landscape_2d(param1_name, param2_name, params, target_data, 
                           current, time_config, n_points=15):
    """
    Plot 2D loss landscape (varying two parameters).
    
    Creates both a 3D surface plot and a 2D contour plot.
    
    Args:
        param1_name (str): First parameter to vary
        param2_name (str): Second parameter to vary
        params (dict): Current parameters
        target_data (dict): Target data
        current (np.ndarray): Input current
        time_config (dict): Time configuration
        n_points (int): Number of points per dimension (total = n_points²)
    """
    # Parameter ranges (±20% of current)
    val1_current = params[param1_name]
    val2_current = params[param2_name]
    
    val1_range = np.linspace(val1_current * 0.8, val1_current * 1.2, n_points)
    val2_range = np.linspace(val2_current * 0.8, val2_current * 1.2, n_points)
    
    # Create grid
    val1_grid, val2_grid = np.meshgrid(val1_range, val2_range)
    loss_grid = np.zeros_like(val1_grid)
    
    print(f"\nEvaluating 2D loss landscape ({n_points}x{n_points} = {n_points**2} points)...")
    print("This may take a moment...")
    
    # Compute loss for each grid point
    v_initial = params['v_rest']
    total_points = n_points * n_points
    for i in range(n_points):
        for j in range(n_points):
            # Progress indicator
            current_point = i * n_points + j + 1
            if current_point % 25 == 0:
                print(f"  Progress: {current_point}/{total_points} points...")
            
            # Modified parameters
            params_modified = params.copy()
            params_modified[param1_name] = val1_grid[i, j]
            params_modified[param2_name] = val2_grid[i, j]
            
            # Simulate and compute loss
            voltage, spikes = simulate_neuron_euler(
                params_modified, time_config, current, v_initial
            )
            simulated = {
                'voltage': voltage,
                'spike_times': spikes,
                'time_config': time_config
            }
            loss_result = compute_combined_loss(simulated, target_data, params_modified)
            loss_grid[i, j] = loss_result['total']
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    min_val1 = val1_grid[min_idx]
    min_val2 = val2_grid[min_idx]
    min_loss = loss_grid[min_idx]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(val1_grid, val2_grid, loss_grid, cmap='viridis', alpha=0.8)
    
    # Mark current and optimal points
    current_loss_idx = np.argmin(np.abs(val1_range - val1_current))
    current_loss_jdx = np.argmin(np.abs(val2_range - val2_current))
    current_loss = loss_grid[current_loss_jdx, current_loss_idx]
    
    ax1.scatter([val1_current], [val2_current], [current_loss], 
               color='red', s=100, label='Current')
    ax1.scatter([min_val1], [min_val2], [min_loss], 
               color='green', s=150, marker='*', label='Optimum')
    ax1.set_xlabel(param1_name, fontsize=11)
    ax1.set_ylabel(param2_name, fontsize=11)
    ax1.set_zlabel('Loss', fontsize=11)
    ax1.set_title('3D Loss Landscape', fontsize=13)
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Subplot 2: Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(val1_grid, val2_grid, loss_grid, levels=15, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)

    ax2.plot(val1_current, val2_current, 'ro', markersize=10, label='Current')
    ax2.plot(min_val1, min_val2, 'g*', markersize=15, label='Optimum')

    ax2.set_xlabel(param1_name, fontsize=12)
    ax2.set_ylabel(param2_name, fontsize=12)
    ax2.set_title('Contour Plot (Loss Levels)', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nCurrent: {param1_name}={val1_current:.2f}, {param2_name}={val2_current:.2f}, Loss={current_loss:.2f}")
    print(f"Optimum: {param1_name}={min_val1:.2f}, {param2_name}={min_val2:.2f}, Loss={min_loss:.2f}")

def main():
    """
    Demonstrate gradient computation and visualization.
    """
    from src.layer1_parameters import get_default_parameters
    
    print("="*60)
    print("LAYER 4: GRADIENT COMPUTATION DEMONSTRATION")
    print("="*60)
    
    # Setup
    print("\n📊 Setting up simulation...")
    time_config = create_time_configuration(dt=0.1, t_total=100.0)
    time = time_config['time']
    current = create_constant_inputs(time, amplitude=18.0)
    
    # Generate target data with "true" parameters
    print("📊 Generating target data...")
    target_data = generate_target_data(current, time_config, noise_level=0.0)
    
    true_params = target_data['params']
    print(f"\nTrue (hidden) parameters:")
    print(f"  tau: {true_params['tau']:.1f} ms")
    print(f"  v_rest: {true_params['v_rest']:.1f} mV")
    print(f"  v_threshold: {true_params['v_threshold']:.1f} mV")
    print(f"  v_reset: {true_params['v_reset']:.1f} mV")
    
    # Start with "wrong" parameters (what we're trying to learn)
    current_params = get_default_parameters()
    print(f"\nCurrent (initial guess) parameters:")
    print(f"  tau: {current_params['tau']:.1f} ms")
    print(f"  v_rest: {current_params['v_rest']:.1f} mV")
    print(f"  v_threshold: {current_params['v_threshold']:.1f} mV")
    print(f"  v_reset: {current_params['v_reset']:.1f} mV")
    
    # Compute current loss
    v_initial = current_params['v_rest']
    voltage_current, spikes_current = simulate_neuron_euler(
        current_params, time_config, current, v_initial
    )
    simulated_current = {
        'voltage': voltage_current,
        'spike_times': spikes_current,
        'time_config': time_config
    }
    loss_current = compute_combined_loss(simulated_current, target_data, current_params)
    print(f"\nCurrent loss: {loss_current['total']:.4f}")
    
    # =================================================================
    # EXAMPLE 1: Compute gradient for ONE parameter
    # =================================================================
    print("\n" + "="*60)
    print("EXAMPLE 1: Gradient for Single Parameter (tau)")
    print("="*60)
    
    print("\nComputing ∂loss/∂tau using finite differences...")
    tau_gradient_result = compute_gradient_finite_diff(
        'tau', current_params, target_data, current, time_config, h=0.01
    )
    
    print(f"\nResults:")
    print(f"  Original tau:     {tau_gradient_result['param_original']:.2f} ms")
    print(f"  Perturbed tau:    {tau_gradient_result['param_perturbed']:.2f} ms")
    print(f"  Original loss:    {tau_gradient_result['loss_original']:.4f}")
    print(f"  Perturbed loss:   {tau_gradient_result['loss_perturbed']:.4f}")
    print(f"  Loss change:      {tau_gradient_result['loss_perturbed'] - tau_gradient_result['loss_original']:.4f}")
    print(f"  Gradient (∂loss/∂tau): {tau_gradient_result['gradient']:.4f}")
    
    if tau_gradient_result['gradient'] > 0:
        print(f"\n💡 Interpretation: Gradient is POSITIVE")
        print(f"   → Increasing tau INCREASES loss")
        print(f"   → Should DECREASE tau to reduce loss")
    else:
        print(f"\n💡 Interpretation: Gradient is NEGATIVE")
        print(f"   → Increasing tau DECREASES loss")
        print(f"   → Should INCREASE tau to reduce loss")
    
    # =================================================================
    # EXAMPLE 2: Compute gradients for ALL parameters
    # =================================================================
    print("\n" + "="*60)
    print("EXAMPLE 2: Gradients for All Parameters")
    print("="*60)
    
    all_gradients = compute_all_gradients_finite_diff(
        current_params, target_data, current, time_config, h=0.01
    )
    
    print_gradient_info(all_gradients)
    
    # =================================================================
    # EXAMPLE 3: Verify gradient direction
    # =================================================================
    print("\n" + "="*60)
    print("EXAMPLE 3: Verify Gradient Directions")
    print("="*60)
    
    print("\nVerifying that gradients point in correct direction...")
    print("(Taking large steps to see clear trends)\n")
    
    for param_name, gradient in all_gradients['gradients'].items():
        verification = verify_gradient_direction(
            param_name, current_params, gradient, 
            target_data, current, time_config, step_size=0.5
        )
        
        print(f"{param_name:12s}: gradient = {gradient:+8.2f}")
        print(f"  Loss original: {verification['loss_original']:.2f}")
        print(f"  Loss (+step):  {verification['loss_plus']:.2f}")
        print(f"  Loss (-step):  {verification['loss_minus']:.2f}")
        print(f"  Verification:  {'✓ CORRECT' if verification['correct'] else '✗ WRONG'}")
        print()
    
    # =================================================================
    # EXAMPLE 4: Visualize 1D loss landscape
    # =================================================================
    print("\n" + "="*60)
    print("EXAMPLE 4: 1D Loss Landscape Visualization")
    print("="*60)
    
    print("\nPlotting loss vs tau...")
    plot_loss_landscape_1d(
        'tau', current_params, target_data, current, time_config, 
        param_range=(15, 30), n_points=20
    )
    
    print("\nPlotting loss vs v_threshold...")
    plot_loss_landscape_1d(
        'v_threshold', current_params, target_data, current, time_config,
        param_range=(-60, -50), n_points=20
    )
    
    # =================================================================
    # EXAMPLE 5: Visualize 2D loss landscape
    # =================================================================
    print("\n" + "="*60)
    print("EXAMPLE 5: 2D Loss Landscape Visualization")
    print("="*60)
    
    print("\nPlotting 2D landscape: tau vs v_threshold")
    print("(This will take a minute...)")
    
    plot_loss_landscape_2d(
        'tau', 'v_threshold', current_params, target_data, 
        current, time_config, n_points=12
    )
    
    # =================================================================
    # EXAMPLE 6: Preview gradient descent step
    # =================================================================
    print("\n" + "="*60)
    print("EXAMPLE 6: Preview One Gradient Descent Step")
    print("="*60)
    
    print("\nWhat if we take ONE step using these gradients?")
    
    learning_rate = 0.5  # Moderate learning rate
    print(f"Learning rate: {learning_rate}")
    
    # Compute updated parameters
    updated_params = current_params.copy()
    print("\nParameter updates:")
    for param_name, gradient in all_gradients['gradients'].items():
        old_value = current_params[param_name]
        new_value = old_value - learning_rate * gradient
        updated_params[param_name] = new_value
        
        change = new_value - old_value
        print(f"  {param_name:12s}: {old_value:7.2f} → {new_value:7.2f}  (Δ = {change:+7.2f})")
    
    # Compute new loss
    voltage_updated, spikes_updated = simulate_neuron_euler(
        updated_params, time_config, current, v_initial
    )
    simulated_updated = {
        'voltage': voltage_updated,
        'spike_times': spikes_updated,
        'time_config': time_config
    }
    loss_updated = compute_combined_loss(simulated_updated, target_data, updated_params)
    
    print(f"\nLoss comparison:")
    print(f"  Before: {all_gradients['loss']:.4f}")
    print(f"  After:  {loss_updated['total']:.4f}")
    print(f"  Change: {loss_updated['total'] - all_gradients['loss']:.4f}")
    
    if loss_updated['total'] < all_gradients['loss']:
        print(f"\n✅ SUCCESS! Loss decreased by gradient descent step!")
        print(f"   This is what Layer 5 will do repeatedly!")
    else:
        print(f"\n⚠️  Loss increased - might need smaller learning rate")
    
    # Compare to true parameters
    print(f"\n" + "="*60)
    print("COMPARISON TO TRUE PARAMETERS")
    print("="*60)
    print(f"\n{'Parameter':<15} {'True':>10} {'Current':>10} {'After Step':>12} {'Error Before':>12} {'Error After':>12}")
    print("-"*75)
    for param_name in ['tau', 'v_rest', 'v_threshold', 'v_reset']:
        true_val = true_params[param_name]
        current_val = current_params[param_name]
        updated_val = updated_params[param_name]
        error_before = abs(current_val - true_val)
        error_after = abs(updated_val - true_val)
        
        print(f"{param_name:<15} {true_val:10.2f} {current_val:10.2f} {updated_val:12.2f} {error_before:12.2f} {error_after:12.2f}")
    
    print("\n" + "="*60)
    print("✅ LAYER 4 COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Gradients tell us which direction to move parameters")
    print("2. Negative gradient → increase parameter to reduce loss")
    print("3. Positive gradient → decrease parameter to reduce loss")
    print("4. Gradient magnitude → how much the parameter matters")
    print("5. Loss landscapes show the optimization terrain")
    print("6. One gradient step moves us toward the optimum")
    print("\nNext: Layer 5 will repeat this process until convergence!")


if __name__ == "__main__":
    main()