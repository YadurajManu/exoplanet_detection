# Real Exoplanet Detection using NASA Kepler Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Download Kepler Dataset
def download_kepler_dataset():
    """
    Download NASA Kepler Exoplanet Dataset
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Kepler Exoplanet Dataset URL
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
    
    # Download file path
    file_path = 'data/kepler_exoplanets.csv'
    
    # Download the dataset
    print("Downloading NASA Kepler Exoplanet Dataset...")
    try:
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Using sample data...")
        return None
    
    return file_path

# Preprocess Kepler Dataset
def preprocess_kepler_data(file_path=None):
    """
    Clean and prepare Kepler Exoplanet Dataset for machine learning
    """
    # If no file, generate sample data
    if file_path is None:
        print("Generating sample exoplanet data...")
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'koi_period': np.random.uniform(0.5, 500, n_samples),
            'koi_time0bk': np.random.uniform(0, 2500, n_samples),
            'koi_depth': np.random.uniform(0, 10000, n_samples),
            'koi_duration': np.random.uniform(0.1, 20, n_samples),
            'koi_insol': np.random.uniform(0, 2000, n_samples),
            'koi_model_snr': np.random.uniform(0, 100, n_samples),
            'koi_tce_plnt_num': np.random.randint(1, 5, n_samples),
            'is_confirmed': np.random.randint(0, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        return df
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Select relevant features for exoplanet detection
    features = [
        'koi_period',  # Orbital Period
        'koi_time0bk',  # Transit Epoch
        'koi_depth',  # Transit Depth
        'koi_duration',  # Transit Duration
        'koi_insol',  # Insolation Flux
        'koi_model_snr',  # Signal-to-Noise Ratio
        'koi_tce_plnt_num',  # Planet Number
    ]
    
    # Determine confirmation status
    df['is_confirmed'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
    
    # Keep only rows with complete data
    df_cleaned = df[features + ['is_confirmed']].dropna()
    
    return df_cleaned

# Visualize Dataset
def visualize_dataset(df):
    """
    Create visualizations of the Exoplanet Dataset
    """
    plt.figure(figsize=(12, 6))
    
    # Distribution of Confirmed vs Unconfirmed Planets
    plt.subplot(1, 2, 1)
    df['is_confirmed'].value_counts().plot(kind='bar')
    plt.title('Confirmed vs Unconfirmed Planets')
    plt.xlabel('Planet Status')
    plt.ylabel('Count')
    
    # Correlation Heatmap
    plt.subplot(1, 2, 2)
    if sns is not None:
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
    else:
        plt.text(0.5, 0.5, 'Seaborn not available\nfor correlation heatmap', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Correlation Visualization Unavailable')
    
    plt.tight_layout()
    plt.savefig('data/exoplanet_dataset_visualization.png')
    plt.close()

# Build Deep Learning Model
def build_exoplanet_model(input_shape):
    """
    Create a deep learning model for exoplanet detection
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
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
    
    return model

# Main Execution
def main():
    # Download Dataset
    dataset_path = download_kepler_dataset()
    
    # Preprocess Data
    df = preprocess_kepler_data(dataset_path)
    
    # Visualize Dataset
    visualize_dataset(df)
    
    # Prepare Features and Target
    X = df.drop('is_confirmed', axis=1)
    y = df['is_confirmed']
    
    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Build and Train Model
    model = build_exoplanet_model(X_train.shape[1])
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate Model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    print("\n🚀 Exoplanet Detection Model Results 🌌")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Exoplanet Detection Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('data/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main()
