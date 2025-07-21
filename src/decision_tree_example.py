import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os

# Create output directory
output_dir = r"/build/dt-example"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create sample dataset with 100 records and 2 parameters
n_samples = 100

# Parameter 1: Continuous variable (gene expression level)
# Using more moderate gene expression variation to reduce its dominance
gene_expression = np.random.normal(5.0, 1.8, n_samples)

# Parameter 2: Binary variable (mutation presence: 0=no, 1=yes)
# Increase mutation probability to have more mutation cases
mutation_present = np.random.binomial(1, 0.5, n_samples)

# Create target variable (binary outcome to predict)
# Disease probability with STRONGER mutation effect and weaker gene expression effect
# Mutation now has much stronger impact on disease probability
linear_combination = 0.6 * gene_expression + 3.5 * mutation_present - 5.5
disease_probability = 1 / (1 + np.exp(-linear_combination))
disease_status = np.random.binomial(1, disease_probability, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'gene_expression': gene_expression,
    'mutation_present': mutation_present,
    'disease_status': disease_status
})

print("Dataset Overview:")
print(f"Total samples: {len(data)}")
print(f"Features: gene_expression (continuous), mutation_present (binary)")
print(f"Target: disease_status (binary)")
print("\nDataset head:")
print(data.head(10))
print("\nDataset summary:")
print(data.describe())

# Show class separation statistics
no_disease_stats = data[data['disease_status'] == 0].groupby('mutation_present')['gene_expression'].describe()
disease_stats = data[data['disease_status'] == 1].groupby('mutation_present')['gene_expression'].describe()
print("\nClass separation analysis by mutation status:")
print("No Disease:")
print(no_disease_stats)
print("\nDisease:")
print(disease_stats)

# Additional analysis to show mutation impact
print("\nMutation impact analysis:")
mutation_disease_rate = data.groupby('mutation_present')['disease_status'].mean()
print(f"Disease rate without mutation: {mutation_disease_rate[0]:.3f}")
print(f"Disease rate with mutation: {mutation_disease_rate[1]:.3f}")
print(f"Mutation increases disease risk by: {(mutation_disease_rate[1] - mutation_disease_rate[0]):.3f}")

# Save dataset
data.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)
print(f"\nDataset saved to {os.path.join(output_dir, 'dataset.csv')}")

# Prepare features and target
X = data[['gene_expression', 'mutation_present']]
y = data['disease_status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train decision tree model
# Using max_depth=3 for interpretability and to avoid overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_split=5, min_samples_leaf=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance
feature_importance = model.feature_importances_
feature_names = ['Gene Expression', 'Mutation Present']

print(f"\nFeature Importance:")
for feature, importance in zip(feature_names, feature_importance):
    print(f"{feature}: {importance:.3f}")

print(f"\nInterpretation:")
if feature_importance[1] > feature_importance[0]:
    print(f"- Mutation presence is the most important feature for predicting disease")
    print(f"- Mutation importance ({feature_importance[1]:.3f}) > Gene expression importance ({feature_importance[0]:.3f})")
else:
    print(f"- Gene expression is still the most important feature")
    print(f"- Gene expression importance ({feature_importance[0]:.3f}) > Mutation importance ({feature_importance[1]:.3f})")
print(f"- Feature importance ratio (Mutation/Gene): {feature_importance[1]/feature_importance[0]:.3f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Decision Tree Results for Bioinformatics Example', fontsize=16)

# Plot 1: Data distribution
mutation_colors = ['lightblue', 'lightcoral']
for mut_status in [0, 1]:
    mask = data['mutation_present'] == mut_status
    scatter = axes[0, 0].scatter(data.loc[mask, 'gene_expression'], 
                               data.loc[mask, 'mutation_present'] + np.random.normal(0, 0.05, mask.sum()),
                               c=data.loc[mask, 'disease_status'], 
                               cmap='RdBu', alpha=0.7, s=50,
                               label=f'Mutation: {mut_status}')

axes[0, 0].set_xlabel('Gene Expression Level')
axes[0, 0].set_ylabel('Mutation Present')
axes[0, 0].set_title('Dataset: Gene Expression vs Mutation Status\n(Color = Disease Status)')
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_yticklabels(['No Mutation', 'Mutation'])
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title('Confusion Matrix')

# Plot 3: Feature Importance
bars = axes[1, 0].bar(feature_names, feature_importance, color=['skyblue', 'lightcoral'])
axes[1, 0].set_ylabel('Importance')
axes[1, 0].set_title('Feature Importance in Decision Tree')
axes[1, 0].grid(True, alpha=0.3)

# Add value labels on bars with percentage
for bar, importance in zip(bars, feature_importance):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{importance:.3f}\n({importance*100:.1f}%)', ha='center', va='bottom')

# Plot 4: Decision Tree Visualization
plot_tree(model, ax=axes[1, 1], feature_names=['Gene_Expr', 'Mutation'], 
          class_names=['No Disease', 'Disease'], filled=True, fontsize=8)
axes[1, 1].set_title('Decision Tree Structure')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_overview.png'), dpi=300, bbox_inches='tight')
print(f"Model overview saved to {os.path.join(output_dir, 'model_overview.png')}")
plt.show()

# Create decision boundary visualization
plt.figure(figsize=(12, 8))

# Create a mesh to plot decision boundaries
h = 0.1
x_min, x_max = X['gene_expression'].min() - 1, X['gene_expression'].max() + 1
y_min, y_max = -0.1, 1.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predictions on the mesh
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(mesh_points)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels=1, alpha=0.4, colors=['lightblue', 'lightcoral'])

# Plot the data points
for mut_status in [0, 1]:
    for disease_status in [0, 1]:
        mask = (data['mutation_present'] == mut_status) & (data['disease_status'] == disease_status)
        if mask.any():
            jitter = np.random.normal(0, 0.02, mask.sum())
            plt.scatter(data.loc[mask, 'gene_expression'], 
                       data.loc[mask, 'mutation_present'] + jitter,
                       c='blue' if disease_status == 0 else 'red',
                       marker='o' if mut_status == 0 else '^',
                       s=60, alpha=0.7, edgecolors='black',
                       label=f'{"No " if disease_status == 0 else ""}Disease, {"No " if mut_status == 0 else ""}Mutation')

plt.xlabel('Gene Expression Level')
plt.ylabel('Mutation Present')
plt.title('Decision Tree Decision Boundaries\nBioinformatics Example: Predicting Disease Status')
plt.yticks([0, 1], ['No Mutation', 'Mutation'])
plt.grid(True, alpha=0.3)

# Create custom legend
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='lightblue', alpha=0.6, label='No Disease Region'),
    mpatches.Patch(color='lightcoral', alpha=0.6, label='Disease Region'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='No Disease'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Disease'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='No Mutation'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Mutation')
]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'decision_boundary.png'), dpi=300, bbox_inches='tight')
print(f"Decision boundary plot saved to {os.path.join(output_dir, 'decision_boundary.png')}")
plt.show()

# Print decision tree rules
print(f"\nDecision Tree Rules:")
tree_rules = export_text(model, feature_names=['gene_expression', 'mutation_present'])
print(tree_rules)

# Save tree rules to file
with open(os.path.join(output_dir, 'decision_rules.txt'), 'w') as f:
    f.write("Decision Tree Rules:\n")
    f.write("==================\n\n")
    f.write(tree_rules)

print(f"\nSample predictions on test set:")
print("Gene_Expr | Mutation | Actual | Predicted | Probability")
print("-" * 55)
for i in range(min(10, len(X_test))):
    print(f"{X_test.iloc[i, 0]:8.2f} | {X_test.iloc[i, 1]:8.0f} | {y_test.iloc[i]:6.0f} | {y_pred[i]:9.0f} | {y_pred_proba[i]:11.3f}")

# Save detailed results
results_df = pd.DataFrame({
    'gene_expression': X_test['gene_expression'].values,
    'mutation_present': X_test['mutation_present'].values,
    'actual_disease': y_test.values,
    'predicted_disease': y_pred,
    'predicted_probability': y_pred_proba
})

# Add model performance metrics and feature importance
performance_metrics = pd.DataFrame({
    'metric': ['accuracy', 'gene_expression_importance', 'mutation_importance', 'mutation_disease_rate_no_mut', 'mutation_disease_rate_mut'],
    'value': [accuracy, feature_importance[0], feature_importance[1], mutation_disease_rate[0], mutation_disease_rate[1]]
})

results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
performance_metrics.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)

print(f"\nResults saved to:")
print(f"- Predictions: {os.path.join(output_dir, 'predictions.csv')}")
print(f"- Model metrics: {os.path.join(output_dir, 'model_metrics.csv')}")
print(f"- Decision rules: {os.path.join(output_dir, 'decision_rules.txt')}")
print(f"- Graphics: model_overview.png, decision_boundary.png")
print(f"- Dataset: dataset.csv")
print(f"- Description: README.md")