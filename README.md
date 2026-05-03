# Optimal Transport Applications & Experiments

A comprehensive project demonstrating practical applications of optimal transport theory using Python. This repository includes a foundational optimal transport library, real-world applications (image color transfer), and experimental comparisons of distance metrics on machine learning tasks.

## 📋 Project Overview

This project showcases optimal transport theory through multiple implementations and applications:

1. **Optimal Transport Fundamentals** (`optimal_transport_demo.py`) - Core algorithms and theory
2. **Image Color Transfer** (`image-color-transfer-ot.ipynb`) - Practical application using Sinkhorn algorithm
3. **MNIST Distance Metrics Experiment** (`mnist-distance-metrics-experiment.ipynb`) - Benchmarking OT vs. traditional metrics
4. **High-Performance Sinkhorn Engine** (`sinkhorn-high-performance-engine.ipynb`) - Production-grade GPU-accelerated implementation with batching, epsilon-scaling, and comprehensive validation

---

## 🚀 Quick Start

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
python optimal_transport_demo.py
```

This generates visualizations and statistics comparing exact OT and Sinkhorn algorithms.

---

## 📁 Project Structure

### Core Script
- **`optimal_transport_demo.py`** - Standalone demonstration of optimal transport algorithms
  - Exact OT using Earth Mover's Distance (EMD)
  - Regularized OT using Sinkhorn-Knopp algorithm
  - Cost matrix computations
  - Wasserstein distance calculations
  - Heatmap visualizations

### Notebooks
- **`image-color-transfer-ot.ipynb`** - Transfer color distributions between images
  - Pixel sampling and preprocessing
  - Sinkhorn algorithm for color transport
  - Image reconstruction and visualization
  - See [IMAGE_COLOR_TRANSFER_OT.md](IMAGE_COLOR_TRANSFER_OT.md) for detailed technical documentation

- **`mnist-distance-metrics-experiment.ipynb`** - Comprehensive distance metric comparison
  - k-NN classification using 4 distance metrics
  - Accuracy and performance benchmarks
  - Sinkhorn regularization parameter tuning
  - Noise robustness testing
  - See [mnist_experiment_results.md](mnist_experiment_results.md) for key findings

- **`sinkhorn-high-performance-engine.ipynb`** - Production-grade GPU-accelerated Sinkhorn optimal transport engine
  - **Three algorithm variants**: Standard log-domain Sinkhorn, fully-vectorized batched Sinkhorn, epsilon-scaling with warm-start
  - **Numerical stability**: Log-domain computation with `torch.logsumexp` prevents underflow/overflow
  - **GPU acceleration**: Automatic CUDA/CPU device selection with 2-4x speedup on batches
  - **Batch processing**: Fully vectorized operations without Python loops over batch dimension
  - **Comprehensive validation**: Transport plan verification with marginal error checking and entropy computation
  - **MNIST benchmarking**: Real-world testing on 784-dimensional distribution problems
  - See [SINKHORN_HIGH_PERFORMANCE_ENGINE.md](SINKHORN_HIGH_PERFORMANCE_ENGINE.md) for complete technical documentation

### Documentation
- **`IMAGE_COLOR_TRANSFER_OT.md`** - Complete technical guide for the color transfer application
  - Algorithm pipeline and mathematical foundations
  - Implementation details and complexity analysis
  - Visual results and use cases
  
- **`mnist_experiment_results.md`** - Experimental results and recommendations
  - Performance metrics and accuracy trade-offs
  - Optimal parameter recommendations
  - Code examples

- **[SINKHORN_HIGH_PERFORMANCE_ENGINE.md](SINKHORN_HIGH_PERFORMANCE_ENGINE.md)** - Complete technical documentation for the GPU-accelerated Sinkhorn engine
  - Architecture and design of three algorithm variants
  - Numerical stability techniques and convergence analysis
  - Performance benchmarks and key observations
  - Validation framework and implementation quality checklist
  - Practical usage examples and recommendations
  - Results tables with iteration counts, timings, and speedups

---

## 🔧 Core Concepts

### Transport Matrix (P)
The solution to the optimal transport problem - a matrix where `P[i,j]` represents the mass transported from source point i to target point j.

**Conservation constraints:**
- Row sums: `P @ 1 = a` (source mass conservation)
- Column sums: `P.T @ 1 = b` (target mass conservation)
- Non-negativity: `P[i,j] ≥ 0`

### Wasserstein Distance
The minimum cost to transform one probability distribution into another:

$$W(a, b) = \min_P \langle P, M \rangle = \sum_{i,j} P[i,j] \times M[i,j]$$

where M is the cost matrix. Lower values indicate more similar distributions.

### Algorithms

#### Exact OT (`ot.emd()`)
- **Pros**: True optimal solution, sparse transport matrices, exact Wasserstein distance
- **Cons**: Slower for large-scale problems (linear programming solver required)
- **Use case**: Offline analysis, high-accuracy requirements

#### Sinkhorn Regularized OT (`ot.sinkhorn()`)
- **Pros**: 5-10x faster than exact OT, entropy regularization, near-optimal accuracy
- **Cons**: Approximate solution
- **Use case**: Real-time applications, large-scale problems

---

## 📊 Key Findings from Experiments

### MNIST Distance Metrics Comparison

| Metric | Accuracy | Speed | Robustness | Best For |
|--------|----------|-------|-----------|----------|
| Euclidean | Fast but lower | ✓✓✓ | Standard | Speed-critical tasks |
| Cosine | Fast but lower | ✓✓✓ | Standard | Large-scale problems |
| **Exact OT (EMD)** | **Highest** | ✗ (slow) | **✓✓✓** | Maximum accuracy |
| **Sinkhorn OT** | **Near-EMD** | **✓✓** | **✓✓✓** | **Best balance** |

**Key takeaway**: Optimal transport metrics significantly outperform traditional distance metrics for image classification, especially on noisy data.

---

## 📝 Code Examples

### Basic Optimal Transport

```python
import numpy as np
import ot

# Define source and target distributions
a = np.array([0.5, 0.5])  # Source histogram
b = np.array([0.3, 0.7])  # Target histogram

# Create cost matrix (e.g., Euclidean distances)
M = np.array([[0.0, 1.0], [1.0, 0.0]])

# Exact OT
P_exact = ot.emd(a, b, M)
wd_exact = np.sum(P_exact * M)  # Wasserstein distance

# Sinkhorn regularized OT
P_sinkhorn = ot.sinkhorn(a, b, M, reg=0.1)
wd_sinkhorn = np.sum(P_sinkhorn * M)

print(f"Exact OT Wasserstein distance: {wd_exact:.4f}")
print(f"Sinkhorn OT Wasserstein distance: {wd_sinkhorn:.4f}")
```

### Image Color Transfer

```python
from PIL import Image
import numpy as np
import ot

# Load images
source_img = Image.open('source.jpg')
target_img = Image.open('target.jpg')

# Color transfer using Sinkhorn algorithm
# (See image-color-transfer-ot.ipynb for complete implementation)
```

---

## 📦 Dependencies

```
pot==0.9.1              # Python Optimal Transport
numpy>=1.20.0           # Numerical computing
matplotlib>=3.3.0       # Visualization
scipy>=1.5.0            # Scientific tools
torch>=1.9.0            # Deep learning (optional, for GPU)
torchvision             # Image utilities
pillow>=8.0.0           # Image processing
scikit-learn            # Machine learning
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🎯 Use Cases

1. **Image Processing**: Color transfer, style transfer, image harmonization
2. **Machine Learning**: Distance metrics for k-NN, distribution matching
3. **Domain Adaptation**: Transfer learning with optimal transport
4. **Data Analysis**: Comparing probability distributions, anomaly detection

---

## 📖 Detailed Documentation

- **[IMAGE_COLOR_TRANSFER_OT.md](IMAGE_COLOR_TRANSFER_OT.md)** - Full technical guide for image color transfer
  - Step-by-step algorithm pipeline
  - Mathematical foundations
  - Implementation details and complexity analysis
  - Visual examples and performance metrics

- **[mnist_experiment_results.md](mnist_experiment_results.md)** - Complete experiment report
  - Methodology and experimental setup
  - Detailed results and analysis
  - Performance recommendations
  - Code examples and usage guide

- **[SINKHORN_IMPLEMENTATION.md](SINKHORN_IMPLEMENTATION.md)** - High-performance Sinkhorn implementation guide
  - Log-domain stabilization techniques
  - GPU acceleration architecture
  - Performance benchmarking framework
  - API reference and usage patterns
  - Numerical stability guarantees

---

## ⚡ High-Performance Sinkhorn Engine

The `sinkhorn-high-performance-engine.ipynb` notebook implements a production-grade optimal transport solver with three algorithmic variants:

### Algorithm Variants

1. **Standard Log-Domain Sinkhorn**
   - Numerically stable computation in log space
   - Marginal error convergence detection
   - Default: max_iter=500, tol=1e-3
   - Use case: Single OT problems, baseline reference

2. **Batched Sinkhorn**
   - Fully vectorized batch processing (no Python loops)
   - Per-batch convergence tracking
   - 2-4x faster than looped version on GPU
   - Use case: Processing multiple OT problems simultaneously

3. **Epsilon-Scaling with Warm-Start**
   - Multi-scale coarse-to-fine refinement
   - Warm-start initialization from coarser scales
   - 30-40% faster wall-clock time vs standard solver
   - Use case: Large problems, high accuracy requirements

### Key Features

**Numerical Stability**
- Log-domain computation prevents underflow/overflow
- `torch.logsumexp` for numerically stable log-sum-exp operations
- Safe clamping before logarithms (1e-16 floor)
- No NaN/Inf even at high problem scales

**GPU Acceleration**
- Automatic CUDA/CPU device detection
- Fully GPU-parallelized batch operations
- Proper device synchronization in benchmarks
- Memory-efficient tensor management

**Validation & Robustness**
- Transport plan validation framework
- Marginal constraint checking (row/column sums)
- NaN/Inf detection and reporting
- Comprehensive error metrics

### Quick Example

```python
import torch
from sinkhorn_high_performance_engine import sinkhorn_log, sinkhorn_batch, sinkhorn_eps_scaling

# Single problem
a = torch.ones(100) / 100
b = torch.ones(100) / 100
M = torch.randn(100, 100).abs()
P, cost, n_iter = sinkhorn_log(a, b, M, reg=0.1, max_iter=500, tol=1e-3, return_stats=True)

# Batch processing
a_batch = torch.ones(16, 100) / 100  # 16 problems
b_batch = torch.ones(16, 100) / 100
M_batch = torch.randn(16, 100, 100).abs()
P_batch, costs, n_iters = sinkhorn_batch(a_batch, b_batch, M_batch, reg=0.1)

# Fast epsilon-scaling
P_eps, cost_eps, total_iters = sinkhorn_eps_scaling(
    a, b, M, reg_schedule=[1.0, 0.5, 0.1], max_iter=200, tol=1e-3
)

print(f"Standard Sinkhorn: {n_iter} iterations, cost {cost:.6f}")
print(f"Batched Sinkhorn: {n_iters.float().mean():.0f} avg iterations")
print(f"Epsilon-Scaling: {total_iters} total iterations, cost {cost_eps:.6f}")
```

### Performance Summary

| Metric | Standard | Batched (B=16) | Epsilon-Scaling |
|--------|----------|----------------|-----------------|
| Iterations | ~170 | ~170 | ~240 |
| Time (100×100) | 0.085s | 0.052s (1.6x faster) | 0.055s (1.5x faster) |
| Validation | ✓ Pass | ✓ Pass | ✓ Pass |
| GPU Speedup | 1.2x | 2.9x | 1.5x |

See [SINKHORN_HIGH_PERFORMANCE_ENGINE.md](SINKHORN_HIGH_PERFORMANCE_ENGINE.md) for complete technical details including convergence analysis, key observations, and implementation quality metrics.

---

## 🔍 Additional Resources

- [Python Optimal Transport Documentation](https://pythonot.github.io/)
- [Wasserstein GANs Paper](https://arxiv.org/abs/1701.07875)
- [Sinkhorn Algorithm Explanation](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Project Information

This project is part of CSE498R coursework, demonstrating practical applications of optimal transport theory in computer science and machine learning.
- Can be numerically less stable

### Sinkhorn OT (`ot.sinkhorn()`)
Pros:
- Much faster (Sinkhorn iterations, no LP solver)
- Can run on GPUs
- More numerically stable
- Scales well

Cons:
- Slightly suboptimal (entropy regularization trade-off)
- Denser solutions (more nonzero entries)
- Needs tuning of regularization parameter

## Example Results

Here's what typical output looks like:

```
Metric                             Exact OT    Sinkhorn OT
Transport Cost                      2.345678      2.412456
Wasserstein Distance                2.345678      2.412456
Sparsity (% zeros)                   75.00%        20.00%
L2 Difference                        0.123456
Relative Cost Difference             2.85%
```

## Tweaking Things

### Adjust problem size and parameters:
```python
n_source = 20        # Increase source points
n_target = 15        # Increase target points
reg_param = 0.05     # Lower = more exact, higher = smoother
```

### Try different distance metrics:
```python
# Instead of Euclidean
from scipy.spatial.distance import cdist

# Use Manhattan distance
M = cdist(X, Y, metric='cityblock')

# Or cosine distance
M = cdist(X, Y, metric='cosine')
```

### Run on GPU with PyTorch:
```python
import torch

a_gpu = torch.tensor(a).cuda()
b_gpu = torch.tensor(b).cuda()
M_gpu = torch.tensor(M).cuda()

P = ot.emd_1d(a_gpu, b_gpu, M_gpu)  # Now on GPU
```

## Useful Resources

- [POT Documentation](https://pythonot.github.io/)
- [Wasserstein Metric on Wikipedia](https://en.wikipedia.org/wiki/Wasserstein_metric)
- [Sinkhorn's Theorem](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)
- Cuturi (2013): "Sinkhorn Distances: Lightspeed Optimal Transport"

## Troubleshooting

### POT not found?
```
pip install pot
```

### LP solver not found?
POT uses scipy by default, but you can install CVXOPT for better performance:
```
pip install cvxopt
```

### Visualization not showing?
On headless servers, save to a file instead by specifying `save_path` in the visualization function, or use:
```python
import matplotlib
matplotlib.use('Agg')
```

### Too slow?
- Use Sinkhorn instead of exact OT for large datasets
- Reduce the problem size
- Enable GPU acceleration

## Quick Tips

1. **Start small** - Use n_source=5, n_target=5 to learn how it works
2. **Always visualize** - Plot the transport matrix to understand what's happening
3. **Check the math** - Verify `P @ 1 ≈ a` and `P.T @ 1 ≈ b`
4. **Pick the right cost** - Make sure your cost matrix M represents what you actually want to measure
5. **Tune Sinkhorn** - Start with reg=0.1, then adjust if needed

## Where to Go From Here

- Experiment with different cost matrices beyond Euclidean distance
- Try it on real data (image distributions, document similarities, etc.)
- Look into more advanced techniques: sliced OT, Gromov-Wasserstein distance
- Use OT in machine learning pipelines (barycenters, alignment)

---

Feel free to modify and experiment!
