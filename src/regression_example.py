import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Create output directory
output_dir = r"/build/regression-example"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("Creating synthetic dataset for systolic blood pressure prediction...")
print("="*60)

# Create synthetic dataset for systolic BP prediction from gene expression
n_samples = 120

# Gene expression levels (representing a hypertension-related gene)
# Range: 0-10 (normalized expression levels)
gene_expression = np.random.uniform(0, 10, n_samples)

# Create a realistic non-linear relationship between gene expression and systolic BP
# Biological rationale: gene expression often has non-linear effects on physiological outcomes

# Base systolic BP (normal range: 90-180 mmHg)
# Non-linear relationship: low expression = normal BP, moderate = slightly elevated, high = hypertension
def true_relationship(x):
    # Complex non-linear relationship with realistic BP values
    base_bp = 100  # Base systolic BP
    linear_component = 3.0 * x  # Linear contribution
    quadratic_component = 0.8 * (x - 5)**2  # Quadratic component (U-shape around x=5)
    interaction_component = 0.2 * x**3 * np.exp(-0.5 * x)  # Complex interaction
    
    return base_bp + linear_component + quadratic_component + interaction_component

# Generate true systolic BP values with realistic noise
systolic_bp_true = true_relationship(gene_expression)

# Add realistic measurement noise (BP measurements have ~5-10 mmHg variability)
measurement_noise = np.random.normal(0, 8, n_samples)
systolic_bp = systolic_bp_true + measurement_noise

# Ensure BP values are within realistic clinical range (90-200 mmHg)
systolic_bp = np.clip(systolic_bp, 90, 200)

# Create DataFrame
data = pd.DataFrame({
    'gene_expression': gene_expression,
    'systolic_bp': systolic_bp,
    'systolic_bp_true': systolic_bp_true  # Keep true values for analysis
})

print(f"Dataset Overview:")
print(f"Total samples: {len(data)}")
print(f"Feature: gene_expression (continuous, 0-10 range)")
print(f"Target: systolic_bp (continuous, mmHg)")
print(f"\nDataset head:")
print(data.head(10))
print(f"\nDataset summary:")
print(data.describe())

# Clinical BP categorization for context
def categorize_bp(bp):
    if bp < 120:
        return 'Normal'
    elif bp < 130:
        return 'Elevated'
    elif bp < 140:
        return 'Stage 1 Hypertension'
    else:
        return 'Stage 2 Hypertension'

data['bp_category'] = data['systolic_bp'].apply(categorize_bp)
bp_distribution = data['bp_category'].value_counts()
print(f"\nBlood Pressure Distribution:")
for category, count in bp_distribution.items():
    print(f"{category}: {count} patients ({count/len(data)*100:.1f}%)")

# Save dataset
data.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)
print(f"\nDataset saved to {os.path.join(output_dir, 'dataset.csv')}")

# Prepare features and target
X = data[['gene_expression']].values
y = data['systolic_bp'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n" + "="*60)
print("REGRESSION MODEL COMPARISON")
print("="*60)

# Dictionary to store model results
models = {}

# 1. Linear Regression
print(f"\n1. LINEAR REGRESSION")
print("-" * 30)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred_linear_train = linear_model.predict(X_train)
y_pred_linear_test = linear_model.predict(X_test)

# Metrics
linear_train_mse = mean_squared_error(y_train, y_pred_linear_train)
linear_test_mse = mean_squared_error(y_test, y_pred_linear_test)
linear_train_r2 = r2_score(y_train, y_pred_linear_train)
linear_test_r2 = r2_score(y_test, y_pred_linear_test)
linear_test_mae = mean_absolute_error(y_test, y_pred_linear_test)

models['Linear'] = {
    'model': linear_model,
    'train_mse': linear_train_mse,
    'test_mse': linear_test_mse,
    'train_r2': linear_train_r2,
    'test_r2': linear_test_r2,
    'test_mae': linear_test_mae,
    'predictions': y_pred_linear_test
}

print(f"Coefficient: {linear_model.coef_[0]:.3f}")
print(f"Intercept: {linear_model.intercept_:.3f}")
print(f"Equation: BP = {linear_model.intercept_:.1f} + {linear_model.coef_[0]:.1f} * gene_expression")
print(f"Train R²: {linear_train_r2:.3f}")
print(f"Test R²: {linear_test_r2:.3f}")
print(f"Test MSE: {linear_test_mse:.3f}")
print(f"Test MAE: {linear_test_mae:.3f} mmHg")

# 2. Polynomial Regression (Degree 2)
print(f"\n2. POLYNOMIAL REGRESSION (Degree 2)")
print("-" * 40)

poly2_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

poly2_pipeline.fit(X_train, y_train)

# Predictions
y_pred_poly2_train = poly2_pipeline.predict(X_train)
y_pred_poly2_test = poly2_pipeline.predict(X_test)

# Metrics
poly2_train_mse = mean_squared_error(y_train, y_pred_poly2_train)
poly2_test_mse = mean_squared_error(y_test, y_pred_poly2_test)
poly2_train_r2 = r2_score(y_train, y_pred_poly2_train)
poly2_test_r2 = r2_score(y_test, y_pred_poly2_test)
poly2_test_mae = mean_absolute_error(y_test, y_pred_poly2_test)

models['Polynomial (Degree 2)'] = {
    'model': poly2_pipeline,
    'train_mse': poly2_train_mse,
    'test_mse': poly2_test_mse,
    'train_r2': poly2_train_r2,
    'test_r2': poly2_test_r2,
    'test_mae': poly2_test_mae,
    'predictions': y_pred_poly2_test
}

print(f"Train R²: {poly2_train_r2:.3f}")
print(f"Test R²: {poly2_test_r2:.3f}")
print(f"Test MSE: {poly2_test_mse:.3f}")
print(f"Test MAE: {poly2_test_mae:.3f} mmHg")

# 3. Polynomial Regression (Degree 3)
print(f"\n3. POLYNOMIAL REGRESSION (Degree 3)")
print("-" * 40)

poly3_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

poly3_pipeline.fit(X_train, y_train)

# Predictions
y_pred_poly3_train = poly3_pipeline.predict(X_train)
y_pred_poly3_test = poly3_pipeline.predict(X_test)

# Metrics
poly3_train_mse = mean_squared_error(y_train, y_pred_poly3_train)
poly3_test_mse = mean_squared_error(y_test, y_pred_poly3_test)
poly3_train_r2 = r2_score(y_train, y_pred_poly3_train)
poly3_test_r2 = r2_score(y_test, y_pred_poly3_test)
poly3_test_mae = mean_absolute_error(y_test, y_pred_poly3_test)

models['Polynomial (Degree 3)'] = {
    'model': poly3_pipeline,
    'train_mse': poly3_train_mse,
    'test_mse': poly3_test_mse,
    'train_r2': poly3_train_r2,
    'test_r2': poly3_test_r2,
    'test_mae': poly3_test_mae,
    'predictions': y_pred_poly3_test
}

print(f"Train R²: {poly3_train_r2:.3f}")
print(f"Test R²: {poly3_test_r2:.3f}")
print(f"Test MSE: {poly3_test_mse:.3f}")
print(f"Test MAE: {poly3_test_mae:.3f} mmHg")

# 4. Polynomial Regression (Degree 4) - to show potential overfitting
print(f"\n4. POLYNOMIAL REGRESSION (Degree 4)")
print("-" * 40)

poly4_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=4, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

poly4_pipeline.fit(X_train, y_train)

# Predictions
y_pred_poly4_train = poly4_pipeline.predict(X_train)
y_pred_poly4_test = poly4_pipeline.predict(X_test)

# Metrics
poly4_train_mse = mean_squared_error(y_train, y_pred_poly4_train)
poly4_test_mse = mean_squared_error(y_test, y_pred_poly4_test)
poly4_train_r2 = r2_score(y_train, y_pred_poly4_train)
poly4_test_r2 = r2_score(y_test, y_pred_poly4_test)
poly4_test_mae = mean_absolute_error(y_test, y_pred_poly4_test)

models['Polynomial (Degree 4)'] = {
    'model': poly4_pipeline,
    'train_mse': poly4_train_mse,
    'test_mse': poly4_test_mse,
    'train_r2': poly4_train_r2,
    'test_r2': poly4_test_r2,
    'test_mae': poly4_test_mae,
    'predictions': y_pred_poly4_test
}

print(f"Train R²: {poly4_train_r2:.3f}")
print(f"Test R²: {poly4_test_r2:.3f}")
print(f"Test MSE: {poly4_test_mse:.3f}")
print(f"Test MAE: {poly4_test_mae:.3f} mmHg")

# Model comparison summary
print(f"\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Test_R2': [models[name]['test_r2'] for name in models.keys()],
    'Test_MSE': [models[name]['test_mse'] for name in models.keys()],
    'Test_MAE': [models[name]['test_mae'] for name in models.keys()],
    'Train_R2': [models[name]['train_r2'] for name in models.keys()]
})

# Sort by Test R²
comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
print(comparison_df.to_string(index=False, float_format='%.3f'))

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} (Test R² = {best_model['test_r2']:.3f})")

# Check for overfitting
print(f"\nOverfitting Analysis:")
for name, model_info in models.items():
    overfitting = model_info['train_r2'] - model_info['test_r2']
    print(f"{name}: Train R² - Test R² = {overfitting:.3f}")
    if overfitting > 0.1:
        print(f"  -> Potential overfitting detected!")
    elif overfitting < 0.02:
        print(f"  -> Good generalization")

# Create visualizations (save only, don't show)
plt.ioff()  # Turn off interactive mode

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Linear vs Polynomial Regression: Systolic Blood Pressure Prediction', fontsize=16)

# Create smooth curve for plotting
X_smooth = np.linspace(0, 10, 300).reshape(-1, 1)
y_true_smooth = true_relationship(X_smooth.flatten())

# Plot 1: Data distribution with true relationship
axes[0, 0].scatter(X, y, alpha=0.6, s=30, color='lightblue', label='Training Data')
axes[0, 0].plot(X_smooth, y_true_smooth, 'r-', linewidth=2, label='True Relationship')
axes[0, 0].set_xlabel('Gene Expression Level')
axes[0, 0].set_ylabel('Systolic BP (mmHg)')
axes[0, 0].set_title('Dataset with True Relationship')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Linear Regression
axes[0, 1].scatter(X_test, y_test, alpha=0.6, s=30, color='lightblue', label='Test Data')
y_pred_smooth_linear = linear_model.predict(X_smooth)
axes[0, 1].plot(X_smooth, y_pred_smooth_linear, 'g-', linewidth=2, label='Linear Regression')
axes[0, 1].plot(X_smooth, y_true_smooth, 'r--', alpha=0.7, label='True Relationship')
axes[0, 1].set_xlabel('Gene Expression Level')
axes[0, 1].set_ylabel('Systolic BP (mmHg)')
axes[0, 1].set_title(f'Linear Regression\n(Test R² = {linear_test_r2:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Best Polynomial Regression
axes[0, 2].scatter(X_test, y_test, alpha=0.6, s=30, color='lightblue', label='Test Data')
y_pred_smooth_best = best_model['model'].predict(X_smooth)
axes[0, 2].plot(X_smooth, y_pred_smooth_best, 'purple', linewidth=2, label=f'{best_model_name}')
axes[0, 2].plot(X_smooth, y_true_smooth, 'r--', alpha=0.7, label='True Relationship')
axes[0, 2].set_xlabel('Gene Expression Level')
axes[0, 2].set_ylabel('Systolic BP (mmHg)')
axes[0, 2].set_title(f'{best_model_name}\n(Test R² = {best_model["test_r2"]:.3f})')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Model Comparison Metrics
models_list = list(models.keys())
test_r2_values = [models[name]['test_r2'] for name in models_list]
test_mae_values = [models[name]['test_mae'] for name in models_list]

x_pos = np.arange(len(models_list))
bars1 = axes[1, 0].bar(x_pos - 0.2, test_r2_values, 0.4, label='Test R²', alpha=0.8)
bars2 = axes[1, 0].bar(x_pos + 0.2, [mae/100 for mae in test_mae_values], 0.4, label='Test MAE (÷100)', alpha=0.8)

axes[1, 0].set_xlabel('Models')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Model Performance Comparison')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(models_list, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars1, test_r2_values):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 5: Residuals for best model
residuals = y_test - best_model['predictions']
axes[1, 1].scatter(best_model['predictions'], residuals, alpha=0.6, s=30)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Predicted Systolic BP (mmHg)')
axes[1, 1].set_ylabel('Residuals (mmHg)')
axes[1, 1].set_title(f'Residual Plot - {best_model_name}')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Predicted vs Actual for best model
axes[1, 2].scatter(y_test, best_model['predictions'], alpha=0.6, s=30)
min_val = min(min(y_test), min(best_model['predictions']))
max_val = max(max(y_test), max(best_model['predictions']))
axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
axes[1, 2].set_xlabel('Actual Systolic BP (mmHg)')
axes[1, 2].set_ylabel('Predicted Systolic BP (mmHg)')
axes[1, 2].set_title(f'Predicted vs Actual - {best_model_name}')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'regression_analysis.png'), dpi=300, bbox_inches='tight')
print(f"\nRegression analysis visualization saved to {os.path.join(output_dir, 'regression_analysis.png')}")
plt.close()

# Create all models comparison plot
plt.figure(figsize=(14, 8))

# Plot all models on the same graph
colors = ['green', 'blue', 'purple', 'orange']
plt.scatter(X, y, alpha=0.4, s=20, color='lightgray', label='All Data')

for i, (name, model_info) in enumerate(models.items()):
    if name == 'Linear':
        y_smooth = linear_model.predict(X_smooth)
    else:
        y_smooth = model_info['model'].predict(X_smooth)
    
    plt.plot(X_smooth, y_smooth, color=colors[i], linewidth=2, 
             label=f'{name} (R² = {model_info["test_r2"]:.3f})')

plt.plot(X_smooth, y_true_smooth, 'r--', linewidth=2, alpha=0.8, label='True Relationship')

plt.xlabel('Gene Expression Level')
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.title('Comparison of Linear and Polynomial Regression Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Model comparison plot saved to {os.path.join(output_dir, 'model_comparison.png')}")
plt.close()

# Save detailed results
test_results = pd.DataFrame({
    'gene_expression': X_test.flatten(),
    'actual_bp': y_test,
    'linear_prediction': y_pred_linear_test,
    'poly2_prediction': y_pred_poly2_test,
    'poly3_prediction': y_pred_poly3_test,
    'poly4_prediction': y_pred_poly4_test
})

# Add residuals
test_results['linear_residual'] = y_test - y_pred_linear_test
test_results['best_model_residual'] = y_test - best_model['predictions']

# Save all results
test_results.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
comparison_df.to_csv(os.path.join(output_dir, 'model_comparison_metrics.csv'), index=False)

print(f"\nResults saved to:")
print(f"- Test predictions: {os.path.join(output_dir, 'test_predictions.csv')}")
print(f"- Model metrics: {os.path.join(output_dir, 'model_comparison_metrics.csv')}")
print(f"- Graphics: regression_analysis.png, model_comparison.png")
print(f"- Dataset: dataset.csv")

# Final summary
print(f"\n" + "="*60)
print("REGRESSION ANALYSIS SUMMARY")
print("="*60)
print(f"+ Dataset: 120 samples of gene expression -> systolic BP prediction")
print(f"+ Best Model: {best_model_name}")
print(f"+ Best Test R²: {best_model['test_r2']:.3f} ({best_model['test_r2']*100:.1f}% variance explained)")
print(f"+ Best Test MAE: {best_model['test_mae']:.1f} mmHg")
print(f"+ Clinical Interpretation:")
print(f"  - Average prediction error: ±{best_model['test_mae']:.1f} mmHg")
print(f"  - Model explains {best_model['test_r2']*100:.1f}% of BP variation")

# Clinical significance
if best_model['test_mae'] < 10:
    print(f"  - Excellent clinical accuracy (error < 10 mmHg)")
elif best_model['test_mae'] < 15:
    print(f"  - Good clinical accuracy (error < 15 mmHg)")
else:
    print(f"  - Moderate clinical accuracy (error > 15 mmHg)")

if best_model['test_r2'] > 0.7:
    print(f"  - Strong predictive relationship")
elif best_model['test_r2'] > 0.5:
    print(f"  - Moderate predictive relationship")
else:
    print(f"  - Weak predictive relationship")

print(f"\nKey Insights:")
print(f"+ Linear regression captures basic trend but misses non-linear patterns")
print(f"+ Polynomial regression better captures complex gene-BP relationships")
print(f"+ Higher degree polynomials risk overfitting with limited data")
print(f"+ Gene expression shows {best_model['test_r2']*100:.1f}% association with systolic BP")

# Turn interactive mode back on
plt.ion()