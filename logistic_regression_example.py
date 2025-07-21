import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os

# Create output directory
output_dir = r"C:\Dev\MLBasics\build\lr-example"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create sample dataset with 100 records and 2 parameters
n_samples = 100

# Parameter 1: Continuous variable (e.g., gene expression level)
# Using a wider range to create more separation between classes
gene_expression = np.random.normal(5.0, 2.5, n_samples)

# Create target variable (binary outcome to predict)
# Let's say we're predicting disease status based on gene expression only
# Higher gene expression increases disease probability with stronger relationship
# Using a steeper coefficient and adjusted threshold for better separation
linear_combination = 1.2 * gene_expression - 6.0
disease_probability = 1 / (1 + np.exp(-linear_combination))
disease_status = np.random.binomial(1, disease_probability, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'gene_expression': gene_expression,
    'disease_status': disease_status
})

print("Dataset Overview:")
print(f"Total samples: {len(data)}")
print(f"Feature: gene_expression (continuous)")
print(f"Target: disease_status (binary)")
print("\nDataset head:")
print(data.head(10))
print("\nDataset summary:")
print(data.describe())

# Show class separation statistics
no_disease_stats = data[data['disease_status'] == 0]['gene_expression'].describe()
disease_stats = data[data['disease_status'] == 1]['gene_expression'].describe()
print("\nClass separation analysis:")
print(f"No Disease - Mean: {no_disease_stats['mean']:.2f}, Std: {no_disease_stats['std']:.2f}")
print(f"Disease    - Mean: {disease_stats['mean']:.2f}, Std: {disease_stats['std']:.2f}")
print(f"Mean difference: {disease_stats['mean'] - no_disease_stats['mean']:.2f}")

# Save dataset
data.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)
print(f"\nDataset saved to {os.path.join(output_dir, 'dataset.csv')}")

# Prepare features and target
X = data[['gene_expression']]
y = data['disease_status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Logistic Regression Results for Bioinformatics Example', fontsize=16)

# Plot 1: Data distribution with larger jitter for better visibility
jitter_y = np.where(data['disease_status'] == 0, 
                    np.random.normal(-0.2, 0.1, len(data)), 
                    np.random.normal(0.2, 0.1, len(data)))
axes[0, 0].scatter(data['gene_expression'], jitter_y, 
                   c=data['disease_status'], cmap='RdBu', alpha=0.7, s=50)
axes[0, 0].set_xlabel('Gene Expression Level')
axes[0, 0].set_ylabel('Disease Status (with jitter)')
axes[0, 0].set_title('Dataset: Gene Expression vs Disease Status')
axes[0, 0].set_yticks([-0.2, 0.2])
axes[0, 0].set_yticklabels(['No Disease', 'Disease'])
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title('Confusion Matrix')

# Plot 3: Prediction probabilities
axes[1, 0].hist(y_pred_proba[y_test == 0], alpha=0.7, bins=15, label='No Disease', color='blue')
axes[1, 0].hist(y_pred_proba[y_test == 1], alpha=0.7, bins=15, label='Disease', color='red')
axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Predicted Probabilities')
axes[1, 0].legend()

# Plot 4: Gene expression distribution by disease status with better bins
no_disease = data[data['disease_status'] == 0]['gene_expression']
with_disease = data[data['disease_status'] == 1]['gene_expression']
axes[1, 1].hist(no_disease, alpha=0.6, bins=12, label='No Disease', color='skyblue', density=True)
axes[1, 1].hist(with_disease, alpha=0.6, bins=12, label='Disease', color='salmon', density=True)
# Add vertical lines for means
axes[1, 1].axvline(no_disease.mean(), color='blue', linestyle='--', alpha=0.8, label=f'No Disease Mean: {no_disease.mean():.1f}')
axes[1, 1].axvline(with_disease.mean(), color='red', linestyle='--', alpha=0.8, label=f'Disease Mean: {with_disease.mean():.1f}')
axes[1, 1].set_xlabel('Gene Expression Level')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Gene Expression Distribution by Disease Status')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_overview.png'), dpi=300, bbox_inches='tight')
print(f"Model overview saved to {os.path.join(output_dir, 'model_overview.png')}")
plt.show()

# Print model coefficients and interpretation
coefficients = model.coef_[0]
print(f"\nModel Coefficients:")
print(f"Intercept: {model.intercept_[0]:.3f}")
print(f"Gene Expression: {coefficients[0]:.3f}")

print(f"\nInterpretation:")
print(f"- Gene Expression coefficient: {coefficients[0]:.3f}")
print(f"  A 1-unit increase in gene expression increases the log-odds of disease by {coefficients[0]:.3f}")

# Create a simple decision boundary visualization
plt.figure(figsize=(10, 8))
h = 0.02  # step size in the mesh

# Create decision boundary for 1D case
gene_range = np.linspace(X['gene_expression'].min() - 1, X['gene_expression'].max() + 1, 300)
gene_range_scaled = scaler.transform(gene_range.reshape(-1, 1))
prob_range = model.predict_proba(gene_range_scaled)[:, 1]

# Plot the probability curve and data points
plt.plot(gene_range, prob_range, 'b-', linewidth=2, label='Predicted Probability')
plt.axhline(y=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')

# Add actual data points with jitter for visibility
jitter = np.where(y == 0, np.random.normal(-0.05, 0.02, len(X)), 
                 np.random.normal(1.05, 0.02, len(X)))
scatter = plt.scatter(X['gene_expression'], jitter, c=y, cmap='RdBu', edgecolors='black', alpha=0.7, s=30)
plt.xlabel('Gene Expression Level')
plt.ylabel('Predicted Probability / Disease Status')
plt.title('Logistic Regression: Gene Expression vs Disease Probability')
plt.grid(True, alpha=0.3)
plt.ylim(-0.15, 1.15)
plt.yticks([0, 0.5, 1.0], ['0', '0.5\n(Threshold)', '1'])

# Add legend with better positioning
handles, labels = scatter.legend_elements()
plt.legend(handles + [plt.Line2D([0], [0], color='blue', linewidth=2), 
                     plt.Line2D([0], [0], color='black', linestyle='--')], 
          ['No Disease', 'Disease', 'Predicted Probability', 'Decision Threshold'],
          loc='center left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'decision_boundary.png'), dpi=300, bbox_inches='tight')
print(f"Decision boundary plot saved to {os.path.join(output_dir, 'decision_boundary.png')}")
plt.show()

print(f"\nSample predictions on test set:")
print("Gene_Expr | Actual | Predicted | Probability")
print("-" * 45)
for i in range(min(10, len(X_test))):
    print(f"{X_test.iloc[i, 0]:8.2f} | {y_test.iloc[i]:6.0f} | {y_pred[i]:9.0f} | {y_pred_proba[i]:11.3f}")

# Save detailed results
results_df = pd.DataFrame({
    'gene_expression': X_test['gene_expression'].values,
    'actual_disease': y_test.values,
    'predicted_disease': y_pred,
    'predicted_probability': y_pred_proba
})

# Add model performance metrics
performance_metrics = pd.DataFrame({
    'metric': ['accuracy', 'intercept', 'gene_expression_coef'],
    'value': [accuracy, model.intercept_[0], coefficients[0]]
})

results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
performance_metrics.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)

print(f"\nResults saved to:")
print(f"- Predictions: {os.path.join(output_dir, 'predictions.csv')}")
print(f"- Model metrics: {os.path.join(output_dir, 'model_metrics.csv')}")
print(f"- Graphics: model_overview.png, decision_boundary.png")
print(f"- Dataset: dataset.csv")
print(f"- Description: README.md")