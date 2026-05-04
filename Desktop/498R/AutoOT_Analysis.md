# AutoOT: Automatic Optimal Transport Solver Selector - Technical Analysis

## Executive Summary

AutoOT is an intelligent system designed to automatically recommend the optimal Optimal Transport (OT) solver for a given dataset based on its characteristics. The system combines rule-based heuristics with meta-learning to provide accurate, interpretable recommendations across diverse data types and structures.

---

## 1. System Architecture Overview

### Core Components

1. **Dataset Feature Extraction Module**
   - Computes 11 key dataset characteristics
   - Enables informed solver selection based on data properties
   - Features include dimensionality, sample count, mass balance, and geometric structure

2. **Rule-Based Recommender (Version 1)**
   - Provides fast, interpretable recommendations
   - Implements 5-level priority hierarchy for solver selection
   - Confidence scores reflect recommendation reliability

3. **Meta-Learning Classifier (Version 2)**
   - Trains RandomForest model on benchmark results
   - Learns empirical solver performance patterns
   - Adapts recommendations based on actual runtime data

4. **Unified API**
   - `recommend_ot()` function combines both approaches
   - Supports hybrid recommendation strategy
   - Returns comprehensive recommendation package with parameters

### Design Rationale

The two-stage approach balances:
- **Interpretability** via rule-based system
- **Accuracy** via data-driven meta-learning
- **Robustness** through ensemble decision-making

---

## 2. Optimal Transport Solvers Implemented

### 2.1 Exact OT (Linear Programming)

**Method:** `ot.emd()` - Earth Mover Distance

**Recommended For:**
- Small datasets (n_samples < 200)
- Low dimensionality (n_dims < 50)
- Balanced mass distributions

**Characteristics:**
- Provides globally optimal solution
- Computationally expensive: O(n³) complexity
- Exact transport matrix computation

**Limitations:**
- Infeasible for large-scale problems
- Sensitive to problem structure
- Requires stable numerical conditioning

---

### 2.2 Sinkhorn Algorithm (Entropic Regularization)

**Method:** `ot.sinkhorn()` - Regularized OT

**Recommended For:**
- Large datasets with balanced masses (n_samples > 200)
- Batch processing scenarios
- When approximate solutions are acceptable

**Characteristics:**
- Entropic regularization reduces complexity to O(n²) per iteration
- Converges faster than exact methods
- Parameter-dependent behavior

**Advantages:**
- Scalable to moderate-sized datasets
- Efficient implementation in POT library
- Provable convergence properties

**Trade-offs:**
- Approximate solution (not globally optimal)
- Regularization parameter tuning required
- Smoothness vs. accuracy trade-off

---

### 2.3 Unbalanced Sinkhorn (Mass-Mismatch Handling)

**Method:** `ot.unbalanced.sinkhorn_unbalanced()` - Relaxed OT

**Recommended For:**
- Datasets with significant mass mismatch (ratio > 1.3 or < 0.77)
- Scenarios with different source/target sample counts
- Flexible coupling requirements

**Characteristics:**
- Relaxes mass conservation constraint
- Introduces mass regularization parameter (reg_m)
- Handles partial transport problems

**Key Parameters:**
- `reg`: Entropic regularization
- `reg_m`: Mass regularization (controls mass conservation relaxation)

**Applications:**
- Partial transport (some mass can be destroyed/created)
- Imbalanced classification
- Domain adaptation with unequal sample counts

---

### 2.4 Gromov-Wasserstein Distance (Geometric Comparison)

**Method:** `ot.gromov_wasserstein()` - Structure-Aware OT

**Recommended For:**
- Graph-structured or high-intra-distance data
- Geometric structure preservation needed
- Data without explicit alignment requirement

**Characteristics:**
- Compares intrinsic geometric structures
- Does not require point-to-point correspondence
- Uses intra-dataset cost matrices (C_source, C_target)

**Advantages:**
- Handles datasets of different dimensions
- Preserves local structure information
- Robust to outliers

**Computational Cost:**
- Higher complexity than Sinkhorn
- Iterative optimization required
- Suitable for moderate-sized datasets

---

### 2.5 Sliced Wasserstein Distance (High-Dimensional)

**Method:** `ot.sliced_wasserstein_distance()` - Dimensional Reduction

**Recommended For:**
- High-dimensional data (n_dims > 50)
- Computationally constrained environments
- When dimensionality curse is a concern

**Characteristics:**
- Projects data onto random 1D subspaces
- Computes distances along projections
- Averages results across projections

**Advantages:**
- Avoids curse of dimensionality
- Linear complexity in dimensions
- Efficient for very high-dimensional data

**Parameters:**
- `n_projections`: Number of random projections (default: 50)
- Higher projections improve accuracy but increase computation

**Limitations:**
- Approximate method
- Quality depends on projection count
- Less interpretable transport plan

---

## 3. Dataset Feature Extraction

### 11 Extracted Features

| Feature | Computation | Purpose |
|---------|------------|---------|
| `n_samples` | Total samples in source + target | Size classification |
| `n_dims` | Feature dimensionality | Dimensionality assessment |
| `avg_sparsity` | Mean proportion of zeros | Data density characterization |
| `mass_ratio` | n_source / n_target | Mass balance detection |
| `mass_balanced` | \|mass_ratio - 1.0\| < 0.3 | Balanced/unbalanced classification |
| `intra_dist` | Mean pairwise distances within datasets | Internal clustering |
| `inter_dist` | Mean distances between source and target | Distribution separation |
| `graph_structure_score` | inter_dist / intra_dist | Structure type indicator |
| `is_graph_structured` | score < 0.5 | Graph detection flag |
| `high_dimension` | n_dims > 50 | Dimensionality flag |
| `high_sample_count` | n_samples > 200 | Size flag |

### Feature Extraction Logic

```
For sparsity: Zero elements counted across both datasets
For mass: Source/target sample count ratio analyzed
For geometry: Pairwise distances sampled (max 30 points for efficiency)
For structure: Ratio of inter-dataset to intra-dataset distances
```

---

## 4. Rule-Based Recommendation Logic

### Decision Hierarchy

```
Priority 1: IF graph_structured
    → Recommend: Gromov-Wasserstein
    → Reason: Structure preservation needed
    → Confidence: 0.9

Priority 2: ELIF mass_mismatch detected
    → Recommend: Unbalanced Sinkhorn
    → Reason: Handle unequal masses
    → Confidence: 0.85
    → Parameters: reg=0.1, reg_m=1.0

Priority 3: ELIF high_sample_count
    → Recommend: Sinkhorn
    → Reason: Scalability priority
    → Confidence: 0.8
    → Parameters: reg=0.1

Priority 4: ELIF high_dimension
    → Recommend: Sliced Wasserstein
    → Reason: Dimensionality curse mitigation
    → Confidence: 0.75
    → Parameters: n_projections=50

Priority 5: ELSE (default)
    → Recommend: Exact OT
    → Reason: Small, balanced, low-dim dataset
    → Confidence: 0.9
```

### Confidence Scoring

Confidence reflects rule reliability:
- **0.9**: Highly reliable recommendations (default case, graph-structured)
- **0.85**: Strong indicators (mass mismatch)
- **0.8**: Common scenario (large datasets)
- **0.75**: Specific constraint (high dimensions)

---

## 5. Benchmark Framework

### Benchmarking Strategy

1. **Solver Testing**: All 5 solvers tested on each dataset
2. **Metrics Collection**: Runtime, transport cost, success flag
3. **Timeout Protection**: Skip if computation exceeds 10 seconds
4. **Failure Handling**: Record unsuccessful attempts with error info

### Results Structure

Each benchmark result captures:
- Dataset identifier
- Solver name
- Execution runtime
- Computed transport cost
- Success/failure status
- Dataset characteristics (for training meta-learner)

### Performance Metrics

- **Runtime**: Execution time in seconds
- **Transport Cost**: OT distance value
- **Success Rate**: Percentage of solvers completing successfully
- **Scalability**: How runtime scales with dataset size

---

## 6. Meta-Learning Classifier

### Model Architecture

**Algorithm:** RandomForestClassifier
- `n_estimators`: 20 trees
- `max_depth`: 5 levels
- `random_state`: 42 (reproducibility)

### Training Data

- **Features**: [n_samples, n_dims] from successful benchmarks
- **Labels**: Fastest solver for each dataset
- **Requirement**: Minimum 2 different solvers for meaningful classification

### Feature Importance Analysis

The model learns relative importance of:
- Sample count vs. dimensionality
- Which factor more strongly predicts optimal solver
- Dataset size sensitivity

### Adaptation Capability

- Retrainable with additional benchmark data
- Enables domain-specific optimization
- Captures empirical performance patterns

---

## 7. Dataset Characteristics

### 7.1 Handwritten Digits (image-like)

**Properties:**
- Source: 100 samples, 64 features
- Target: 100 samples, 64 features
- Type: High-dimensional image data
- Balanced masses

**Expected Solver:** Sliced Wasserstein or Exact OT

---

### 7.2 Wine Dataset (tabular)

**Properties:**
- Source: 50 samples, 13 features
- Target: 50 samples, 13 features
- Type: Chemical properties
- Balanced masses, moderate dimensionality

**Expected Solver:** Exact OT or Sinkhorn

---

### 7.3 Synthetic Blobs (clustered)

**Properties:**
- Source: 150 samples, 10 features
- Target: 150 samples, 10 features
- Type: Synthetically generated clusters
- Balanced masses, well-separated structure

**Expected Solver:** Sinkhorn (moderate size) or Gromov-Wasserstein (structure)

---

### 7.4 Unbalanced Masses (mass mismatch)

**Properties:**
- Source: 100 samples, 8 features
- Target: 50 samples, 8 features
- Type: Intentional mass imbalance (2:1 ratio)
- Unbalanced masses

**Expected Solver:** Unbalanced Sinkhorn

---

## 8. Evaluation Framework

### Evaluation Metrics

1. **Recommendation Accuracy**
   - Percentage of recommendations matching fastest solver
   - Measures alignment with empirical performance

2. **Solver Performance Comparison**
   - Recommended vs. actual fastest solver
   - Recommended vs. best quality solver
   - Runtime analysis

3. **Decision Logic Validation**
   - Verify priority hierarchy matches empirical results
   - Identify edge cases and boundary conditions
   - Assess confidence score calibration

### Result Analysis

**Key Questions Addressed:**
- Does rule-based logic match empirical optima?
- Where does meta-learning improve recommendations?
- Which dataset characteristics most strongly predict solver choice?
- Are there consistent patterns across dataset types?

---

## 9. Key Observations

### 9.1 Solver Performance Patterns

**Exact OT:**
- Optimal for small datasets but scales poorly
- Provides ground truth for solution quality
- Limited to n_samples < 100-200 practically

**Sinkhorn:**
- Most versatile solver for moderate-sized problems
- Good balance between speed and accuracy
- Parameter-sensitive convergence

**Unbalanced Sinkhorn:**
- Specialized for mass mismatch scenarios
- Extends applicability to imbalanced problems
- Trade-off between mass conservation and transport cost

**Gromov-Wasserstein:**
- Unique capability for structure comparison
- Indispensable for graph/network data
- Higher computational cost justified by structure preservation

**Sliced Wasserstein:**
- Essential for high-dimensional scalability
- Quality improves with projection count
- Practical for real-world high-dim applications

### 9.2 Feature Importance

**Primary Drivers:**
1. Mass balance → strongest indicator for solver selection
2. Sample count → determines exact vs. approximate methods
3. Dimensionality → triggers dimensional reduction techniques
4. Geometric structure → determines structure-aware methods

**Secondary Factors:**
- Sparsity patterns
- Distance distributions
- Dataset size ratios

### 9.3 System Design Trade-offs

**Rule-Based Approach:**
- ✓ Interpretable and fast
- ✓ Works with limited data
- ✗ Cannot capture complex patterns
- ✗ May miss optimal solutions

**Meta-Learning Approach:**
- ✓ Learns empirical patterns
- ✓ Adapts to domain characteristics
- ✗ Requires sufficient training data
- ✗ Less interpretable

**Hybrid Strategy:**
- ✓ Combines benefits of both approaches
- ✓ Maintains interpretability as fallback
- ✓ Improves accuracy when meta-learner available
- ✓ Degrades gracefully with limited data

---

## 10. Result Analysis

### 10.1 Recommendation Distribution

Expected patterns across datasets:
- **Digits** (high-dim): Sliced Wasserstein or Exact
- **Wine** (balanced, tabular): Exact OT or Sinkhorn
- **Blobs** (moderate size): Sinkhorn or Gromov-Wasserstein
- **Unbalanced** (mass mismatch): Unbalanced Sinkhorn

### 10.2 Performance Metrics Summary

| Aspect | Expected Finding |
|--------|------------------|
| Runtime scalability | Sliced >> Sinkhorn ≈ GW > Unbalanced > Exact |
| Solution quality | Exact > Unbalanced > Gromov ≈ Sinkhorn > Sliced |
| Memory efficiency | Sliced > Sinkhorn > GW > Unbalanced ≈ Exact |
| Stability | GW ≈ Sinkhorn > Unbalanced > Exact > Sliced |

### 10.3 Critical Success Factors

1. **Accurate Feature Extraction**
   - Directly impacts rule-based decisions
   - Determines meta-learner training quality
   - Essential for all subsequent stages

2. **Benchmark Completeness**
   - Ensures all solvers evaluated fairly
   - Provides ground truth for meta-learner
   - Enables confidence score calibration

3. **Parameter Tuning**
   - Solver parameters strongly affect outcomes
   - Recommend using defaults as baseline
   - Domain-specific tuning may improve results

4. **Timeout Management**
   - Prevents long-running computations
   - Eliminates pathological cases
   - Critical for practical deployment

---

## 11. Practical Applications

### 11.1 Use Cases

1. **Automated Algorithm Selection**
   - Select solver based on data characteristics
   - Reduce manual tuning requirements
   - Enable reproducible recommendations

2. **Comparative Analysis**
   - Understand why certain solvers fit particular data
   - Learn feature-to-solver mappings
   - Identify optimization opportunities

3. **Production Systems**
   - Real-time solver selection
   - Adaptive recommendations
   - Performance monitoring

4. **Research and Development**
   - Benchmark new OT solver implementations
   - Validate theoretical predictions
   - Explore algorithm combinations

### 11.2 Implementation Considerations

- **Data Preprocessing**: Ensure consistent scaling across source/target
- **Feature Scaling**: StandardScaler applied to all datasets
- **Reproducibility**: Fixed random seed (42) for consistent results
- **Error Handling**: Graceful degradation on solver failures
- **Documentation**: All parameters and decisions logged

---

## 12. Limitations and Considerations

### Current Limitations

1. **Limited Solver Diversity**
   - 5 core solvers covered
   - Other OT variants not included (e.g., semi-relaxed, unregularized)

2. **Dataset Size Constraints**
   - Tested on small-to-moderate datasets
   - Exact OT timeout (10s) may not suit all applications
   - Very large-scale problems may need specialized handling

3. **Feature Set**
   - 11 features may not capture all relevant properties
   - Domain-specific features might improve recommendations

4. **Meta-Learner Dependence**
   - Requires sufficient training data diversity
   - May not generalize to novel dataset types

### Mitigation Strategies

- Extend feature set with domain-specific properties
- Include additional solvers (e.g., entropic GW, sliced GW)
- Increase benchmark dataset diversity
- Implement adaptive threshold tuning
- Regular model retraining with new data

---

## 13. Conclusion

AutoOT provides a principled, data-driven approach to Optimal Transport solver selection. By combining interpretable rule-based heuristics with empirical meta-learning, the system achieves robust recommendations across diverse data types. The architecture balances accuracy, interpretability, and computational efficiency, making it suitable for both research and practical applications.

### Strengths
- Systematic approach to a complex selection problem
- Two-stage design provides flexibility and robustness
- Comprehensive feature analysis enables understanding
- Extensible architecture for future solvers

### Future Directions
- Expand solver coverage with additional OT variants
- Enhance feature extraction with domain-specific metrics
- Implement online learning for continuous improvement
- Develop visualizations for recommendation explainability
- Create specialized variants for specific application domains

---

**Document Version:** 1.0  
**Last Updated:** 2026  
**Technical Framework:** Python Optimal Transport (POT) library
