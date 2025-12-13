# Neuron Learning Journey

A step-by-step implementation of differentiable programming for computational neuroscience, built from first principles.

## Project Overview

This project demonstrates **differentiable programming** by building a complete pipeline to learn neuron parameters from simulated "experimental" data. Starting with basic definitions and culminating in gradient-based optimization, this serves as both an educational resource and a practical tool.


##  Architecture

The project is organized into 5 layers, each building on the previous:
```
Layer 1: Entity Definition
  ├── Parameters (tau, v_rest, v_threshold, v_reset)
  ├── Input current patterns
  ├── Voltage state initialization
  └── Time representation

Layer 2: Forward Simulation
  ├── Single time-step dynamics (Euler integration)
  ├── Full time-loop simulation
  ├── Spike detection and reset
  └── Visualization tools

Layer 3: Loss Functions
  ├── Target data generation
  ├── Voltage MSE (full and sub-threshold)
  ├── Spike-based metrics (count, timing, rate)
  └── Combined multi-objective loss

Layer 4: Gradient Computation
  ├── Finite difference gradients
  ├── Gradient verification
  ├── Loss landscape visualization (1D & 2D)
  └── Numerical differentiation methods

Layer 5: Learning
  ├── Gradient descent optimization
  ├── Momentum-based acceleration
  ├── Parameter convergence tracking
  └── Learned vs target comparison
```

##  Quick Start

### Installation
```bash
# Install dependencies
pip install numpy matplotlib


### Run a Complete Example
```python
# Run the full learning demo
python src/layer5_learning.py

# This will:
# 1. Generate synthetic target data
# 2. Start with wrong parameters
# 3. Learn through gradient descent
# 4. Show parameter convergence
# 5. Compare learned vs target neuron
```

### Use Individual Components
```python
from src.layer1_parameters import get_default_parameters
from src.layer2_full_simulation import simulate_neuron_euler
from src.layer3_loss_functions import compute_combined_loss
from src.layer4_gradients import compute_all_gradients_finite_diff
from src.layer5_learning import gradient_descent

# Setup
params = get_default_parameters()
time_config = create_time_configuration(dt=0.1, t_total=100.0)
current = create_constant_input(time_config['time'], amplitude=15.0)

# Simulate
voltage, spikes = simulate_neuron_euler(params, time_config, current, v_initial=-70)

# Learn
result = gradient_descent(params, target_data, current, time_config)
learned_params = result['final_params']
```

##  Example Results

### Learning Curve
The loss decreases exponentially as parameters converge:
```
Iteration   0: Loss = 45.2345
Iteration  10: Loss = 12.4521
Iteration  20: Loss =  3.1234
Iteration  30: Loss =  0.4521
Iteration  40: Loss =  0.0234
 Converged!
```

### Parameter Recovery

| Parameter    | True Value | Initial Guess | Learned Value | Error  |
|--------------|------------|---------------|---------------|--------|
| tau          | 25.0 ms    | 20.0 ms       | 24.87 ms      | 0.13   |
| v_threshold  | -52.0 mV   | -55.0 mV      | -52.14 mV     | 0.14   |

## Key Concepts

### Differentiable Programming

This project demonstrates **differentiable programming**: building programs where you can compute gradients with respect to parameters. This enables:

- Automatic parameter tuning
- Data-driven model building
- Integration of domain knowledge with learning

### The Learning Loop
```python
for iteration in range(max_iterations):
    # Forward: simulate with current parameters
    voltage = simulate(params)
    
    # Loss: measure error
    loss = compute_loss(voltage, target)
    
    # Backward: compute gradients
    gradients = compute_gradients(params)
    
    # Update: improve parameters
    params = params - learning_rate * gradients
```

### Handling Discontinuities

Spike resets are discontinuous (non-differentiable). We handle this through:

1. **Numerical gradients** - treat simulation as black box
2. **Sub-threshold loss** - only optimize smooth regions  
3. **Spike-based metrics** - count/timing losses

## Project Structure
```
neuron-learning-journey/
├── src/
│   ├── __init__.py
│   ├── layer1_parameters.py          # Parameter definitions
│   ├── layer1_input.py                # Input current patterns
│   ├── layer1_voltage_state.py        # Voltage initialization
│   ├── layer1_time_representation.py  # Time configuration
│   ├── layer2_single_step.py          # Single timestep simulation
│   ├── layer2_full_simulation.py      # Complete time loop
│   ├── layer2_visualization.py        # Voltage plotting
│   ├── layer3_target_data.py          # Target generation
│   ├── layer3_loss_functions.py       # Error metrics
│   ├── layer4_gradients.py            # Gradient computation
│   └── layer5_learning.py             # Optimization loop
├── notebooks/                          # Jupyter experiments
├── docs/                               # Documentation
├── README.md                           # This file
├── requirements.txt                    # Dependencies
└── .gitignore
```

## The Neuron Model

### Leaky Integrate-and-Fire (LIF)

The model implements the LIF equation:
```
τ dV/dt = -(V - V_rest) + I
```

With spike mechanism:
```
if V ≥ V_threshold:
    emit spike
    V ← V_reset
```

### Parameters

- **τ (tau)**: Membrane time constant (ms) - controls leak rate
- **V_rest**: Resting potential (mV) - equilibrium voltage
- **V_threshold**: Spike threshold (mV) - firing trigger
- **V_reset**: Reset potential (mV) - post-spike hyperpolarization

