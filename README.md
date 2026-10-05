#  ML Models from Scratch
 
> **Build intuition, not just libraries.** Implementations 
> of fundamental ML algorithms from the ground up using 
> NumPy — with detailed explanations of the math and 
> reasoning behind every line of code.

##  Overview
 
This repository contains **clean, well-documented implementations** of core machine learning algorithms. Rather than treating them as black boxes, each model is built to help you understand:
- **Why** the algorithm works
- **How** gradient descent, entropy, and bootstrapping actually function
- **When** to use each model in practice
Perfect for interview prep, ML deepening, or teaching others.
 
---

##  Implemented Models
 
### **Supervised Learning**
 
| Model | Purpose | Key Concepts | Status     |
|-------|---------|--------------|------------|
| **Linear Regression** | Continuous prediction | Gradient Descent, MSE Loss |  Complete |
| **Logistic Regression** | Binary classification | Sigmoid, Cross-Entropy Loss, Gradient Descent |  Complete  |
| **Decision Tree** | Classification/Regression | Information Gain, Entropy, Recursive Splitting | Complete   |
| **K-Nearest Neighbors** | Non-parametric classification | Euclidean Distance, Majority Voting | Complete   |
| **Naive Bayes** | Probabilistic classification | Bayes' Theorem, Laplace Smoothing, Log-Sum Trick | Complete   |

### **Ensemble Methods**
 
| Model | Purpose | Key Concepts | Status     |
|-------|---------|--------------|------------|
| **Random Forest** | Robust classification | Bagging, Bootstrap Sampling, Feature Randomness |  Complete |
| **XGBoost** | Gradient boosted trees | (Scikit-learn wrapper + hyperparameter tuning demo) |  Complete  |

### **Unsupervised Learning**
 
| Model | Purpose | Key Concepts | Status     |
|-------|---------|--------------|------------|
| **K-Means Clustering** | Partitional clustering | EM Loop, Centroid Initialization, Convergence | Complete   |
| **PCA** | Dimensionality reduction | Variance Preservation, Standardization |  Complete  |
| **Support Vector Machines** | Classification with margin | (Scikit-learn wrapper + grid search demo) |  Complete |
 
---

## 📁 Repository Structure
 
```
.
├── Linear_Regression.py          # y = mx + b with gradient descent
├── Logistic_Regression.py        # Binary classification via sigmoid
├── Decision_Tree.py              # Entropy-based recursive splits
├── KNN.py                        # K-nearest neighbors from scratch
├── Naive_Bayes.py               # Probabilistic classifier + Laplace smoothing
├── K_Means.py                    # EM-loop based clustering
├── Random_Forest.py              # Bagging + decision trees ensemble
├── PCA.py                        # Variance-preserving dimensionality reduction
├── SVM.py                        # Grid search tuning demo
├── XGBoost.py                    # Imbalanced classification + feature importance
├── Transformers.py               # (Placeholder for future work)
└── README.md                     # This file
```

**Each file is self-contained** with:
- Clear class definitions
- Inline comments explaining mathematical concepts
- A `__main__` block with working examples and expected outputs
---

## 🚀 Quick Start
 
### Prerequisites
```bash
pip install numpy scikit-learn xgboost pandas
```

### Run Any Model
```bash
# Decision Tree example
python Decision_Tree.py
 
# Output:
# --- Model Inference Verification ---
# Test Sample 1 Prediction (Expected 0): 0
# Test Sample 2 Prediction (Expected 1): 1
```

Each file includes:
1. **Model implementation** (the `__main__` block is optional)
2. **Toy dataset** for quick testing
3. **Expected outputs** for validation
---

## 💡 Key Concepts Explained
 
### **Gradient Descent** (Linear/Logistic Regression)
- How weights update via partial derivatives
- Learning rate and epoch tuning
- Loss visualization every 100 iterations

### **Information Gain & Entropy** (Decision Trees)
- Splitting criteria: parent entropy vs child entropy
- Best split selection: greedy feature search
- Stopping conditions: max depth, min samples
### **Bootstrap Aggregating** (Random Forest)
- Sampling with replacement for diverse trees
- Feature randomness at each split
- Majority voting for final predictions