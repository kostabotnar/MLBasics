import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os

# Create output directory
output_dir = r"C:\Dev\MLBasics\build\svm-example"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create sample dataset with 100 records - Extended with age parameter
n_samples = 100

# Parameter 1: Continuous variable (gene expression level)
# Using the same improved parameters as the logistic regression example
gene_expression = np.random.normal(5.0, 2.5, n_samples)

# Parameter 2: Age (continuous variable, realistic range for medical study)
# Age range: 25-75 years with normal distribution
age = np.random.normal(50.0, 12.0, n_samples)
age = np.clip(age, 25, 75)  # Clip to realistic age range

# Create target variable (binary outcome to predict)
# Disease status based on BOTH gene expression and age
# Higher gene expression and older age increase disease probability
linear_combination = 1.2 * gene_expression + 0.05 * age - 8.5
disease_probability = 1 / (1 + np.exp(-linear_combination))
disease_status = np.random.binomial(1, disease_probability, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'gene_expression': gene_expression,
    'age': age,
    'disease_status': disease_status
})

print("Dataset Overview:")
print(f"Total samples: {len(data)}")
print(f"Features: gene_expression (continuous), age (continuous)")
print(f"Target: disease_status (binary)")
print("\nDataset head:")
print(data.head(10))
print("\nDataset summary:")
print(data.describe())

# Show class separation statistics for both features
print("\nClass separation analysis:")
for feature in ['gene_expression', 'age']:
    no_disease_stats = data[data['disease_status'] == 0][feature].describe()
    disease_stats = data[data['disease_status'] == 1][feature].describe()
    print(f"\n{feature.upper()}:")
    print(f"  No Disease - Mean: {no_disease_stats['mean']:.2f}, Std: {no_disease_stats['std']:.2f}")
    print(f"  Disease    - Mean: {disease_stats['mean']:.2f}, Std: {disease_stats['std']:.2f}")
    print(f"  Mean difference: {disease_stats['mean'] - no_disease_stats['mean']:.2f}")

# Save dataset
data.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)
print(f"\nDataset saved to {os.path.join(output_dir, 'dataset.csv')}")

# Prepare features and target
X = data[['gene_expression', 'age']]
y = data['disease_status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM models with different kernels
svm_models = {
    'Linear': SVC(kernel='linear', probability=True, random_state=42),
    'RBF': SVC(kernel='rbf', probability=True, random_state=42, gamma='scale'),
    'Polynomial': SVC(kernel='poly', degree=2, probability=True, random_state=42, gamma='scale')
}

model_results = {}

for name, model in svm_models.items():
    print(f"\nTraining {name} SVM...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'support_vectors': model.support_vectors_ if hasattr(model, 'support_vectors_') else None,
        'n_support_vectors': model.n_support_[0] + model.n_support_[1] if hasattr(model, 'n_support_') else 0
    }
    
    print(f"{name} SVM Accuracy: {accuracy:.3f}")
    print(f"Number of support vectors: {model_results[name]['n_support_vectors']}")
    if hasattr(model, 'coef_') and model.coef_ is not None:
        print(f"Coefficients: {model.coef_[0]}")
    if hasattr(model, 'intercept_'):
        print(f"Intercept: {model.intercept_[0]:.3f}")

# Select best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
best_model = model_results[best_model_name]
print(f"\nBest model: {best_model_name} SVM with accuracy: {best_model['accuracy']:.3f}")

# Detailed evaluation of best model
print(f"\n{best_model_name} SVM Classification Report:")
print(classification_report(y_test, best_model['predictions']))

# Create visualizations (save only, don't show)
plt.ioff()  # Turn off interactive mode

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Support Vector Machine Results for Bioinformatics Example', fontsize=16)

# Plot 1: 2D scatter plot of both features
scatter = axes[0, 0].scatter(data['gene_expression'], data['age'], 
                            c=data['disease_status'], cmap='RdBu', alpha=0.7, s=50)
axes[0, 0].set_xlabel('Gene Expression Level')
axes[0, 0].set_ylabel('Age (years)')
axes[0, 0].set_title('Dataset: Gene Expression vs Age\n(Color = Disease Status)')
axes[0, 0].grid(True, alpha=0.3)
# Add colorbar for clarity
cbar = plt.colorbar(scatter, ax=axes[0, 0])
cbar.set_label('Disease Status')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['No Disease', 'Disease'])

# Plot 2: Confusion Matrix for best model
cm = confusion_matrix(y_test, best_model['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title(f'Confusion Matrix - {best_model_name} SVM')

# Plot 3: Model comparison
model_names = list(model_results.keys())
accuracies = [model_results[name]['accuracy'] for name in model_names]
colors = ['lightblue', 'lightcoral', 'lightgreen']
bars = axes[1, 0].bar(model_names, accuracies, color=colors[:len(model_names)])
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('SVM Kernel Comparison')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(True, alpha=0.3)

# Add accuracy labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

# Plot 4: Feature correlation and support vectors
# Show correlation between features and highlight support vectors
axes[1, 1].scatter(data['gene_expression'], data['age'], 
                   c=data['disease_status'], cmap='RdBu', alpha=0.6, s=30)

# Mark support vectors if available (show in original feature space)
if best_model['support_vectors'] is not None:
    sv_original = scaler.inverse_transform(best_model['support_vectors'])
    axes[1, 1].scatter(sv_original[:, 0], sv_original[:, 1], 
                      s=100, facecolors='none', edgecolors='red', linewidth=2,
                      label=f'Support Vectors (n={best_model["n_support_vectors"]})')

axes[1, 1].set_xlabel('Gene Expression Level')
axes[1, 1].set_ylabel('Age (years)')
axes[1, 1].set_title('Feature Space + Support Vectors')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_overview.png'), dpi=300, bbox_inches='tight')
print(f"Model overview saved to {os.path.join(output_dir, 'model_overview.png')}")
plt.close()

# Create 2D decision boundary visualization for best model
plt.figure(figsize=(12, 10))

# Create a mesh to plot the decision boundaries
h = 0.1  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 5, X.iloc[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predictions on the mesh
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)
Z = best_model['model'].predict(mesh_points_scaled)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels=1, alpha=0.4, colors=['lightblue', 'lightcoral'])
plt.contour(xx, yy, Z, levels=[0.5], colors=['black'], linestyles=['--'], linewidths=2)

# Add actual data points
scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='RdBu', 
                     edgecolors='black', alpha=0.7, s=50)

# Mark support vectors
if best_model['support_vectors'] is not None:
    sv_original = scaler.inverse_transform(best_model['support_vectors'])
    plt.scatter(sv_original[:, 0], sv_original[:, 1], 
               s=150, facecolors='none', edgecolors='red', linewidth=3,
               label=f'Support Vectors (n={best_model["n_support_vectors"]})')

plt.xlabel('Gene Expression Level')
plt.ylabel('Age (years)')
plt.title(f'{best_model_name} SVM: 2D Decision Boundary\nGene Expression vs Age')
plt.grid(True, alpha=0.3)

# Add legend
handles, labels = scatter.legend_elements()
plt.legend(handles + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                                 markeredgecolor='red', markersize=10, linewidth=2),
                     plt.Line2D([0], [0], color='black', linestyle='--')], 
          ['No Disease', 'Disease', 'Support Vectors', 'Decision Boundary'],
          loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'decision_boundary.png'), dpi=300, bbox_inches='tight')
print(f"Decision boundary plot saved to {os.path.join(output_dir, 'decision_boundary.png')}")
plt.close()

# Create 2D kernel comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('SVM Kernel Comparison: 2D Decision Boundaries', fontsize=16)

for idx, (name, result) in enumerate(model_results.items()):
    ax = axes[idx]
    
    # Create mesh for this kernel
    Z_kernel = result['model'].predict(mesh_points_scaled)
    Z_kernel = Z_kernel.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z_kernel, levels=1, alpha=0.4, colors=['lightblue', 'lightcoral'])
    ax.contour(xx, yy, Z_kernel, levels=[0.5], colors=['black'], linestyles=['--'], linewidths=1.5)
    
    # Add data points
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='RdBu', 
                        edgecolors='black', alpha=0.7, s=30)
    
    # Mark support vectors if available
    if result['support_vectors'] is not None:
        sv_original = scaler.inverse_transform(result['support_vectors'])
        ax.scatter(sv_original[:, 0], sv_original[:, 1], 
                  s=80, facecolors='none', edgecolors='red', linewidth=2)
    
    ax.set_xlabel('Gene Expression Level')
    ax.set_ylabel('Age (years)')
    ax.set_title(f'{name} Kernel\nAcc: {result["accuracy"]:.3f}, SVs: {result["n_support_vectors"]}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kernel_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Kernel comparison plot saved to {os.path.join(output_dir, 'kernel_comparison.png')}")
plt.close()

print(f"\nSample predictions on test set ({best_model_name} SVM):")
print("Gene_Expr | Age | Actual | Predicted | Probability")
print("-" * 55)
for i in range(min(10, len(X_test))):
    print(f"{X_test.iloc[i, 0]:8.2f} | {X_test.iloc[i, 1]:3.0f} | {y_test.iloc[i]:6.0f} | {best_model['predictions'][i]:9.0f} | {best_model['probabilities'][i]:11.3f}")

# Save detailed results
results_df = pd.DataFrame({
    'gene_expression': X_test.iloc[:, 0].values,
    'age': X_test.iloc[:, 1].values,
    'actual_disease': y_test.values,
    'predicted_disease': best_model['predictions'],
    'predicted_probability': best_model['probabilities'],
    'best_model': [best_model_name] * len(X_test)
})

# Create comprehensive model metrics
all_model_metrics = []
for name, result in model_results.items():
    all_model_metrics.append({
        'model': name,
        'accuracy': result['accuracy'],
        'n_support_vectors': result['n_support_vectors']
    })

model_metrics_df = pd.DataFrame(all_model_metrics)

# Save support vectors information
support_vectors_info = pd.DataFrame({
    'model': [name for name in model_results.keys()],
    'n_support_vectors': [model_results[name]['n_support_vectors'] for name in model_results.keys()],
    'support_vector_ratio': [model_results[name]['n_support_vectors']/len(X_train) for name in model_results.keys()]
})

results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
model_metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
support_vectors_info.to_csv(os.path.join(output_dir, 'support_vectors_info.csv'), index=False)

print(f"\nResults saved to:")
print(f"- Predictions: {os.path.join(output_dir, 'predictions.csv')}")
print(f"- Model metrics: {os.path.join(output_dir, 'model_metrics.csv')}")
print(f"- Support vectors info: {os.path.join(output_dir, 'support_vectors_info.csv')}")
print(f"- Graphics: model_overview.png, decision_boundary.png, kernel_comparison.png")
print(f"- Dataset: dataset.csv")
print(f"- Description: README.md")

# Turn interactive mode back on
plt.ion()