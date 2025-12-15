"""
===============================================================================
DIFFERENTIABLE HODGKIN-HUXLEY - COMPREHENSIVE BIOPHYSICAL ANALYSIS
===============================================================================
Publication-quality results showing complete neural dynamics
===============================================================================
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib.pyplot as plt

# [Previous code: sigmoid, transform_params, alpha/beta functions, HH equations stay the same]

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -500, 500)))

def transform_params(theta):
    theta_Na, theta_K, theta_L = theta
    gNa = 1.0 + 199.0 * sigmoid(theta_Na)
    gK  = 1.0 +  99.0 * sigmoid(theta_K)
    gL  = 0.01 + 0.99 * sigmoid(theta_L)
    return gNa, gK, gL

def inverse_transform(gNa, gK, gL):
    x_Na = jnp.clip((gNa - 1.0) / (200.0 - gNa), 1e-6, 1e6)
    x_K = jnp.clip((gK - 1.0) / (100.0 - gK), 1e-6, 1e6)
    x_L = jnp.clip((gL - 0.01) / (1.0 - gL), 1e-6, 1e6)
    return jnp.array([jnp.log(x_Na), jnp.log(x_K), jnp.log(x_L)])

def alpha_m(v): return 0.1 * (v + 40.0) / (1.0 - jnp.exp(-(v + 40.0) / 10.0) + 1e-7)
def beta_m(v): return 4.0 * jnp.exp(-(v + 65.0) / 18.0)
def alpha_h(v): return 0.07 * jnp.exp(-(v + 65.0) / 20.0)
def beta_h(v): return 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))
def alpha_n(v): return 0.01 * (v + 55.0) / (1.0 - jnp.exp(-(v + 55.0) / 10.0) + 1e-7)
def beta_n(v): return 0.125 * jnp.exp(-(v + 65.0) / 80.0)

def hh_simulate_with_gating(theta, I_input, dt=0.01):
    """H-H simulation that returns gating variables"""
    gNa, gK, gL = transform_params(theta)
    ENa, EK, EL = 50.0, -77.0, -54.387
    C = 1.0

    v0 = -65.0
    m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
    h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
    n0 = alpha_n(v0) / (alpha_n(v0) + beta_n(v0))

    state0 = jnp.array([v0, m0, h0, n0])

    def step(state, I):
        v, m, h, n = state

        dm = alpha_m(v) * (1.0 - m) - beta_m(v) * m
        dh = alpha_h(v) * (1.0 - h) - beta_h(v) * h
        dn = alpha_n(v) * (1.0 - n) - beta_n(v) * n

        INa = gNa * m**3 * h * (v - ENa)
        IK  = gK * n**4 * (v - EK)
        IL  = gL * (v - EL)

        dv = (I - INa - IK - IL) / C

        v_new = jnp.clip(v + dt * dv, -150.0, 100.0)
        m_new = m + dt * dm
        h_new = h + dt * dh
        n_new = n + dt * dn

        state_new = jnp.array([v_new, m_new, h_new, n_new])
        return state_new, (v_new, m_new, h_new, n_new, INa, IK, IL)

    _, outputs = jax.lax.scan(step, state0, I_input)
    v_trace, m_trace, h_trace, n_trace, INa_trace, IK_trace, IL_trace = outputs
    
    return v_trace, m_trace, h_trace, n_trace, INa_trace, IK_trace, IL_trace

# ============================================================================
# DATA & TRAINING (same as before)
# ============================================================================

def generate_data(seed=42):
    np.random.seed(seed)
    dt = 0.01
    T = 150.0
    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps)

    gNa_true, gK_true, gL_true = 120.0, 36.0, 0.3
    params_true = inverse_transform(gNa_true, gK_true, gL_true)

    I = np.zeros(n_steps)
    I[(times > 20) & (times < 40)] = 10.0
    I[(times > 80) & (times < 100)] = 10.0
    I = jnp.array(I)

    v_clean, _, _, _, _, _, _ = hh_simulate_with_gating(params_true, I, dt)
    v_noisy = v_clean + 0.3 * jnp.array(np.random.randn(len(v_clean)))
    
    print("\n" + "="*70)
    print("STEP 1: Generate Synthetic Data")
    print("="*70)
    print(f"Generated {len(times)} timepoints with TWO input pulses")
    print(f"True: gNa={gNa_true}, gK={gK_true}, gL={gL_true}")
    print(f"Voltage range: [{jnp.min(v_noisy):.2f}, {jnp.max(v_noisy):.2f}] mV")

    return times, v_noisy, I, params_true

def make_loss(v_target, I_input, dt=0.01):
    def loss(theta):
        v_sim, _, _, _, _, _, _ = hh_simulate_with_gating(theta, I_input, dt)
        return jnp.mean((v_sim - v_target) ** 2)
    return loss

class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.95, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, theta, grads):
        if self.m is None:
            self.m = jnp.zeros_like(theta)
            self.v = jnp.zeros_like(theta)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        theta = theta - self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        return theta

def train(v_target, I_input, params_true):
    print("\n" + "="*70)
    print("STEP 2: Learn Parameters with ADAM Optimizer")
    print("="*70)
    
    loss_fn = make_loss(v_target, I_input, dt=0.01)
    grad_fn = jit(grad(loss_fn))

    theta = jnp.array([jnp.log(0.5), jnp.log(0.2), jnp.log(0.01)])

    adam = AdamOptimizer(learning_rate=0.001, beta1=0.95, beta2=0.999)
    epochs = 10000
    losses = []
    grad_norms = []

    gNa_t, gK_t, gL_t = transform_params(params_true)
    
    print(f"\nTrue parameters: gNa={gNa_t:.2f}, gK={gK_t:.2f}, gL={gL_t:.4f}")

    best_loss = float('inf')
    best_theta = theta.copy()
    no_improve = 0
    
    for ep in range(epochs):
        grads = grad_fn(theta)
        grad_norm = jnp.linalg.norm(grads)
        grads = jnp.clip(grads, -1.0, 1.0)
        theta = adam.step(theta, grads)
        
        loss = loss_fn(theta)
        losses.append(float(loss))
        grad_norms.append(float(grad_norm))
        
        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 300:
            print(f"Early stopping at epoch {ep}")
            theta = best_theta
            break

        if ep % 200 == 0 or ep == epochs - 1:
            gNa, gK, gL = transform_params(theta)
            errors = [100*abs(gNa-gNa_t)/gNa_t, 100*abs(gK-gK_t)/gK_t, 100*abs(gL-gL_t)/gL_t]
            print(f"Epoch {ep:4d} | Loss {loss:8.4f} | gNa Δ{errors[0]:5.1f}% | gK Δ{errors[1]:5.1f}% | gL Δ{errors[2]:5.1f}%")

    return theta, losses

# ============================================================================
# COMPREHENSIVE BIOPHYSICAL PLOTTING
# ============================================================================

def plot_comprehensive_results(times, v_target, I_input, theta_learned, params_true, losses):
    """Publication-quality comprehensive biophysical analysis"""
    
    # Get learned parameters
    gNa_learned, gK_learned, gL_learned = transform_params(theta_learned)
    gNa_true, gK_true, gL_true = transform_params(params_true)
    
    # Simulate to get ALL components
    v_final, m_final, h_final, n_final, INa_final, IK_final, IL_final = hh_simulate_with_gating(theta_learned, I_input)
    
    errors = [100*abs(gNa_learned-gNa_true)/gNa_true, 
              100*abs(gK_learned-gK_true)/gK_true,
              100*abs(gL_learned-gL_true)/gL_true]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # ===== ROW 1: Voltage Trace & Convergence =====
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(times, v_target, 'k-', linewidth=2.5, label='Target (experimental)', alpha=0.8)
    ax1.plot(times, v_final, 'g-', linewidth=2.5, label='Learned model', alpha=0.8)
    ax1.fill_between(times, v_target, v_final, alpha=0.1, color='blue')
    ax1.axvspan(20, 40, alpha=0.05, color='yellow')
    ax1.axvspan(80, 100, alpha=0.05, color='yellow')
    ax1.set_ylabel('Voltage (mV)', fontsize=11, weight='bold')
    ax1.set_title('A) Voltage Trace: Target vs Learned', fontsize=11, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-85, 45])
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.semilogy(losses, 'b-', linewidth=2.5)
    ax2.fill_between(range(len(losses)), losses, alpha=0.2, color='blue')
    ax2.set_ylabel('Loss (MSE)', fontsize=11, weight='bold')
    ax2.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax2.set_title('B) Training Convergence', fontsize=11, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    x = np.arange(3)
    width = 0.35
    ax3.bar(x - width/2, [gNa_true, gK_true, gL_true*100], width, label='True',
            color='navy', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.bar(x + width/2, [gNa_learned, gK_learned, gL_learned*100], width, label='Learned',
            color='green', alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, err in enumerate(errors):
        color = 'green' if err < 5 else 'orange' if err < 10 else 'red'
        ax3.text(i, max([gNa_true, gK_true, gL_true*100][i], [gNa_learned, gK_learned, gL_learned*100][i]) * 1.05,
                f'{err:.1f}%', ha='center', fontsize=9, weight='bold', color=color)
    ax3.set_ylabel('Conductance', fontsize=11, weight='bold')
    ax3.set_title('C) Parameter Recovery', fontsize=11, weight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['gNa', 'gK', 'gL'])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== ROW 2: Gating Variables =====
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(times, m_final, 'r-', linewidth=2.5, label='m (Na+ activation)')
    ax4.set_ylabel('Gating Value', fontsize=11, weight='bold')
    ax4.set_title('D) Na+ Activation (m)', fontsize=11, weight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(times, h_final, 'b-', linewidth=2.5, label='h (Na+ inactivation)')
    ax5.set_ylabel('Gating Value', fontsize=11, weight='bold')
    ax5.set_title('E) Na+ Inactivation (h)', fontsize=11, weight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(times, n_final, 'g-', linewidth=2.5, label='n (K+ activation)')
    ax6.set_ylabel('Gating Value', fontsize=11, weight='bold')
    ax6.set_title('F) K+ Activation (n)', fontsize=11, weight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # ===== ROW 3: Ion Currents =====
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(times, INa_final, 'r-', linewidth=2.5)
    ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax7.fill_between(times, 0, INa_final, alpha=0.2, color='red')
    ax7.set_ylabel('Current (μA/cm²)', fontsize=11, weight='bold')
    ax7.set_xlabel('Time (ms)', fontsize=11, weight='bold')
    ax7.set_title('G) I_Na (Sodium)', fontsize=11, weight='bold')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(times, IK_final, 'g-', linewidth=2.5)
    ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax8.fill_between(times, 0, IK_final, alpha=0.2, color='green')
    ax8.set_ylabel('Current (μA/cm²)', fontsize=11, weight='bold')
    ax8.set_xlabel('Time (ms)', fontsize=11, weight='bold')
    ax8.set_title('H) I_K (Potassium)', fontsize=11, weight='bold')
    ax8.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(times, IL_final, 'gray', linewidth=2.5)
    ax9.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax9.fill_between(times, 0, IL_final, alpha=0.2, color='gray')
    ax9.set_ylabel('Current (μA/cm²)', fontsize=11, weight='bold')
    ax9.set_xlabel('Time (ms)', fontsize=11, weight='bold')
    ax9.set_title('I) I_L (Leak)', fontsize=11, weight='bold')
    ax9.grid(True, alpha=0.3)
    
    fig.suptitle('Hodgkin-Huxley Model: Differentiable Parameter Learning from Electrophysiology', 
                fontsize=14, weight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('hh_parameter_learning.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved as 'hh_parameter_learning.png'")
    try:
        plt.show()
    except:
        pass
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"gNa: {gNa_true:.2f} → {gNa_learned:.2f} ({errors[0]:.1f}% error)")
    print(f"gK:  {gK_true:.2f} → {gK_learned:.2f} ({errors[1]:.1f}% error)")
    print(f"gL:  {gL_true:.4f} → {gL_learned:.4f} ({errors[2]:.1f}% error)")
    print("="*70 + "\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("JAX devices:", jax.devices())

    print("DIFFERENTIABLE HODGKIN-HUXLEY PARAMETER LEARNING")
    print("="*70)

    times, v_target, I_input, params_true = generate_data()
    theta_learned, losses = train(v_target, I_input, params_true)
    plot_comprehensive_results(times, v_target, I_input, theta_learned, params_true, losses)

if __name__ == "__main__":
    main()