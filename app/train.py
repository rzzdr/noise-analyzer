import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from app.NoiseAnalyzer import NoiseAnalyzer, TARGET_CLASSES

# Training parameters
BATCH_SIZE = 10240
EPOCHS = 300

def plot_training_curves(history):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy curves
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Loss curves
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def train_model(analyzer, features, labels, labels_raw):
    """Train the model with callbacks"""
    print("Splitting dataset...")
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp, labels_train, labels_temp = train_test_split(
        features, labels, labels_raw, test_size=0.3, random_state=42, stratify=labels_raw
    )
    
    # Second split: 15% validation, 15% test from the 30% temp
    X_val, X_test, y_val, y_test, labels_val, labels_test = train_test_split(
        X_temp, y_temp, labels_temp, test_size=0.5, random_state=42, stratify=labels_temp
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    analyzer.model = analyzer.create_model()
    print(analyzer.model.summary())
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        TensorBoard(log_dir='./logs', histogram_freq=1)
    ]
    
    print("Training model...")
    # Reshape features to remove single channel dimension for Conv1D
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
    
    history = analyzer.model.fit(
        X_train_reshaped, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_reshaped, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    test_loss, test_accuracy = analyzer.model.evaluate(X_test_reshaped, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred = analyzer.model.predict(X_test_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=TARGET_CLASSES, labels=range(len(TARGET_CLASSES)), 
                              zero_division=0))
    
    # Per-class F1 scores
    f1_scores = f1_score(y_true_classes, y_pred_classes, average=None, labels=range(len(TARGET_CLASSES)), zero_division=0)
    for i, class_name in enumerate(TARGET_CLASSES):
        print(f"F1 Score for {class_name}: {f1_scores[i]:.4f}")
    
    # Plot training curves
    plot_training_curves(history)
    
    # Save model
    analyzer.model.save('models/noise_classifier_model.h5')
    print("Model saved as 'models/noise_classifier_model.h5'")
    
    # Save normalization parameters
    np.savez('models/model_params.npz', 
            scaler_mean=analyzer.scaler_mean, 
            scaler_std=analyzer.scaler_std)
    print("Normalization parameters saved as 'models/model_params.npz'")

    return history

def main():
    print("4-Class Audio Classification System - Training Mode")
    print("(Silence detection handled separately by VAD)")
    print("="*55)
    
    # Initialize analyzer
    analyzer = NoiseAnalyzer()
    
    # Check if ESC-50 dataset exists
    if not os.path.exists(analyzer.dataset_path):
        print(f"ESC-50 dataset not found at {analyzer.dataset_path}")
        print("Please download ESC-50 dataset from: https://github.com/karolpiczak/ESC-50")
        print("Extract it to the current directory as 'data/ESC-50-master'")
        return
    
    try:
        # Load and prepare dataset
        features, labels, labels_raw = analyzer.load_esc50_dataset()
        
        # Train model
        history = train_model(analyzer, features, labels, labels_raw)
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("Files created:")
        print("  - models/best_model.h5 (best model during training)")
        print("  - models/noise_classifier_model.h5 (final model)")
        print("  - model_params.npz (normalization parameters)")
        print("  - training_curves.png (training plots)")
        print("  - confusion_matrix.png (evaluation results)")
        print("\nYou can now use main.py for real-time prediction!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main()
