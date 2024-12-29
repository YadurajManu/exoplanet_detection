# Comprehensive Exoplanet Detection Script
# Leveraging Machine Learning for NASA Kepler Dataset Analysis

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Configuration
OUTPUT_DIR = '/Users/sujeetkumarsingh/Desktop/Chatbot/exoplanet_detection/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(filepath):
    """Load and preprocess Kepler exoplanet dataset."""
    data = pd.read_csv(filepath)
    
    # Select most relevant features for exoplanet detection
    features = [
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 
        'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 
        'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_smet'
    ]
    
    # Create binary target variable
    data['is_confirmed'] = (data['koi_disposition'] == 'CONFIRMED').astype(int)
    
    # Select features and target
    X = data[features]
    y = data['is_confirmed']
    
    # Drop rows with missing values
    X_cleaned = X.dropna()
    y_cleaned = y[X_cleaned.index]
    
    print(f"Total samples: {len(X_cleaned)}")
    print(f"Confirmed exoplanets: {y_cleaned.sum()}")
    print(f"Unconfirmed exoplanets: {len(y_cleaned) - y_cleaned.sum()}")
    
    return X_cleaned, y_cleaned


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest Classifier."""
    pipeline = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        class_weight='balanced',
        random_state=42
    )
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    print("\n🌟 Random Forest Classification Report:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return pipeline


def train_deep_learning(X_train, X_test, y_train, y_test):
    """Train Deep Learning Neural Network."""
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Predict and evaluate
    y_pred_proba = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\n🌌 Deep Learning Classification Report:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model


def main():
    """Main execution function."""
    # Load data
    filepath = '/Users/sujeetkumarsingh/Desktop/Chatbot/exoplanet_detection/data/kepler_exoplanet_data.csv'
    X, y = load_data(filepath)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    train_random_forest(X_train, X_test, y_train, y_test)
    train_deep_learning(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
