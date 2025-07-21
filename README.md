# MLBasics: Machine Learning Examples for Bioinformatics

A comprehensive collection of machine learning examples tailored for bioinformatics education and presentations. This repository demonstrates various ML algorithms using realistic biological datasets and medical applications.

## 🧬 Overview

This project provides hands-on examples of fundamental machine learning algorithms applied to bioinformatics problems, including disease prediction, patient clustering, and biomarker analysis. Each example includes detailed explanations, visualizations, and clinical interpretations.

## 📊 Examples Included

### 1. **Logistic Regression** (`build/lr-example/`)
- **Application**: Disease status prediction from gene expression
- **Dataset**: 100 samples with single continuous feature
- **Key Features**: 
  - Sigmoid probability curves
  - Decision boundary visualization
  - High accuracy (80%) with clear interpretation
- **Files**: `logistic_regression_example.py`, visualizations, detailed README

### 2. **Decision Trees** (`build/dt-example/`)
- **Application**: Disease prediction with gene expression and mutation status
- **Dataset**: 100 samples with 2 features (1 continuous, 1 binary)
- **Key Features**:
  - Interpretable if-then rules
  - Feature importance analysis (mutation 75%, gene expression 25%)
  - Tree structure visualization
- **Files**: `decision_tree_example.py`, decision rules, visualizations

### 3. **Support Vector Machines** (`build/svm-example/`)
- **Application**: Disease prediction using gene expression and age
- **Dataset**: 100 samples with 2 continuous features
- **Key Features**:
  - Multiple kernel comparison (Linear, RBF, Polynomial)
  - 2D decision boundaries
  - Excellent performance (90% accuracy)
  - Support vector identification
- **Files**: `svm_example.py`, kernel comparisons, decision boundaries

### 4. **K-Means Clustering** (`build/knn-example/`)
- **Application**: Patient stratification based on gene expression and age
- **Dataset**: Same as SVM example for cross-method comparison
- **Key Features**:
  - Unsupervised patient grouping
  - Cluster purity analysis (97.3%)
  - KNN neighborhood analysis
  - Comparison with known disease status
- **Files**: `knn_clustering_example.py`, cluster visualizations

### 5. **DBSCAN Clustering** (`build/dbscan-example/`)
- **Application**: Density-based patient clustering with outlier detection
- **Dataset**: Synthetic dataset optimized for DBSCAN (150 samples)
- **Key Features**:
  - Non-spherical cluster detection
  - Automatic outlier identification
  - Parameter optimization using k-distance plots
  - Superior performance vs K-Means (ARI: 0.606 vs 0.529)
- **Files**: `dbscan_clustering_example.py`, parameter exploration

### 6. **Linear vs Polynomial Regression** (`build/regression-example/`)
- **Application**: Systolic blood pressure prediction from gene expression
- **Dataset**: 120 samples with realistic clinical BP values
- **Key Features**:
  - Model complexity comparison (Linear vs Polynomial degrees 2-4)
  - Overfitting detection and prevention
  - Clinical accuracy assessment (±6.4 mmHg error)
  - Excellent variance explanation (60.2%)
- **Files**: `regression_example.py`, model comparisons, residual analysis

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Clone and Run
```bash
git clone https://github.com/[username]/MLBasics.git
cd MLBasics

# Run any example
python logistic_regression_example.py
python decision_tree_example.py
python svm_example.py
python knn_clustering_example.py
python dbscan_clustering_example.py
python regression_example.py
```

## 📁 Project Structure

```
MLBasics/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── logistic_regression_example.py    # Single-feature classification
├── decision_tree_example.py          # Interpretable tree models
├── svm_example.py                    # 2D SVM with multiple kernels
├── knn_clustering_example.py         # K-means patient clustering
├── dbscan_clustering_example.py      # Density-based clustering
├── regression_example.py             # Linear vs polynomial regression
└── build/                            # Generated results and documentation
    ├── lr-example/                   # Logistic regression results
    ├── dt-example/                   # Decision tree results
    ├── svm-example/                  # SVM results
    ├── knn-example/                  # K-means clustering results
    ├── dbscan-example/               # DBSCAN clustering results
    └── regression-example/           # Regression analysis results
```

## 🎯 Educational Goals

### For Students
- **Hands-on Experience**: Working with realistic bioinformatics datasets
- **Visual Learning**: Comprehensive plots and visualizations for each algorithm
- **Clinical Context**: Understanding how ML applies to medical problems
- **Method Comparison**: Direct performance comparisons between algorithms

### For Instructors
- **Ready-to-Use Examples**: Complete code with detailed explanations
- **Presentation Materials**: High-quality visualizations and documentation
- **Progressive Complexity**: Examples build from simple to advanced concepts
- **Real-World Applications**: Biologically relevant scenarios and interpretations

## 📊 Key Results Summary

| Algorithm | Dataset | Accuracy/Performance | Key Insight |
|-----------|---------|---------------------|-------------|
| Logistic Regression | Gene expression → Disease | 80% accuracy | Simple yet effective for binary classification |
| Decision Trees | Gene + Mutation → Disease | 80% accuracy | Highly interpretable rules, mutation dominance |
| SVM | Gene + Age → Disease | 90% accuracy | Excellent 2D separation with RBF kernel |
| K-Means | Gene + Age → Clusters | 97.3% purity | Good patient stratification |
| DBSCAN | Synthetic clusters | ARI: 0.606 | Superior for non-spherical patterns |
| Polynomial Regression | Gene → Blood Pressure | R²: 0.602, ±6.4 mmHg | Non-linear relationships crucial |

## 🔬 Biological Applications

### Clinical Decision Support
- **Risk Stratification**: Identifying high-risk patients
- **Biomarker Discovery**: Finding predictive genetic markers
- **Treatment Response**: Predicting therapeutic outcomes
- **Disease Subtyping**: Discovering patient subgroups

### Research Applications
- **Genomics**: Gene expression analysis and pathway discovery
- **Personalized Medicine**: Tailoring treatments to individual profiles
- **Drug Discovery**: Identifying therapeutic targets
- **Population Health**: Understanding disease patterns

## 📈 Visualizations Included

Each example generates comprehensive visualizations:
- **Data distributions** with clinical annotations
- **Model performance** comparisons and metrics
- **Decision boundaries** and classification regions
- **Feature importance** and coefficient interpretations
- **Residual analysis** and model diagnostics
- **Clinical interpretation** guides

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new bioinformatics examples
- Improve existing documentation
- Suggest additional clinical applications
- Report issues or bugs

## 📚 References and Further Reading

- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Bioinformatics Applications**: Clinical genomics and precision medicine
- **Statistical Learning**: "The Elements of Statistical Learning" by Hastie et al.
- **Python for Data Science**: "Python Data Science Handbook" by VanderPlas

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgments

- Built for bioinformatics education and presentations
- Designed with realistic clinical scenarios and medical interpretations
- Optimized for both learning and teaching purposes

---

**Perfect for**: Bioinformatics courses, ML workshops, clinical research presentations, and hands-on learning experiences.