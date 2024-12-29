# Exoplanet Classification Machine Learning Model

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report
)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Step 2: Load and Prepare the Dataset
# We'll use the simulated data from our previous exploration
np.random.seed(42)

# Simulate more detailed exoplanet data
data = {
    'planet_radius': np.random.normal(10, 5, 1000),  # Planet radius in Earth radii
    'orbital_period': np.random.lognormal(2, 1, 1000),  # Orbital period in days
    'equilibrium_temp': np.random.normal(300, 100, 1000),  # Temperature in Kelvin
    'stellar_magnitude': np.random.normal(10, 2, 1000),  # Stellar magnitude
    'is_confirmed': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # Confirmation status
}

df = pd.DataFrame(data)

# Step 3: Prepare Features and Target Variable
# Features are the input variables
X = df[['planet_radius', 'orbital_period', 'equilibrium_temp', 'stellar_magnitude']]
# Target is the variable we want to predict
y = df['is_confirmed']

# Step 4: Split the Data into Training and Testing Sets
# This helps us evaluate how well our model performs on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% of data for testing
    random_state=42  # Ensures reproducibility
)

# Step 5: Feature Scaling
# Standardize features to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the Machine Learning Model
# We'll use Random Forest Classifier - a powerful algorithm for classification
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate Model Performance
print("Model Performance Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Visualize Feature Importances
plt.figure(figsize=(10, 6))
feature_importance = model.feature_importances_
features = X.columns
indices = np.argsort(feature_importance)

plt.title('Feature Importances for Exoplanet Classification')
plt.barh(range(len(indices)), feature_importance[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('data/feature_importance.png')
plt.close()

# Step 10: Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('data/confusion_matrix.png')
plt.close()

# Bonus: Example Prediction Function
def predict_exoplanet(radius, period, temp, magnitude):
    # Prepare input data
    input_data = np.array([[radius, period, temp, magnitude]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    print(f"\nPrediction for Exoplanet:")
    print(f"Radius: {radius} Earth radii")
    print(f"Orbital Period: {period} days")
    print(f"Equilibrium Temperature: {temp} Kelvin")
    print(f"Stellar Magnitude: {magnitude}")
    print(f"\nIs Confirmed Exoplanet: {'Yes' if prediction[0] == 1 else 'No'}")
    print(f"Confirmation Probability: {probability[0][1]*100:.2f}%")

# Example usage of prediction function
predict_exoplanet(
    radius=12.5, 
    period=15.3, 
    temp=350, 
    magnitude=9.5
)

print("\nModel training and evaluation complete!")
print("Visualizations saved in 'data/' directory.")
