import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sounddevice as sd
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Audio parameters
SAMPLE_RATE = 16000
FRAME_LENGTH = 400  # 25ms at 16kHz
HOP_LENGTH = 160    # 10ms at 16kHz
N_FFT = 512
N_MELS = 40
DURATION = 1.0      # 1 second clips
N_FRAMES = 100      # Fixed frame count for consistency

# Model parameters
BATCH_SIZE = 2048
EPOCHS = 200
LEARNING_RATE = 0.001

# Class mapping from ESC-50 to target classes
ESC50_TO_TARGET = {
    # Silence - low energy or actual silence
    'breathing': 'Silence',
    'silence': 'Silence',
    'wind': 'Silence',
    
    # Whispering - closest approximations
    'sneezing': 'Whispering',
    
    # Typing sounds
    'keyboard_typing': 'Typing',
    'mouse_click': 'Typing',
    'writing': 'Typing',
    
    # Phone/Alert sounds
    'phone': 'Phone_ringing',
    'alarm_clock': 'Phone_ringing',
    'clock_alarm': 'Phone_ringing',
    'clock_tick': 'Phone_ringing',
    
    # Loud talking - various vocal/loud sounds
    'laughing': 'Loud_talking',
    'coughing': 'Loud_talking',
    'crying_baby': 'Loud_talking',
    'snoring': 'Loud_talking',
    'children_playing': 'Loud_talking',
    'conversation': 'Loud_talking',
    'footsteps': 'Loud_talking',
    'clapping': 'Loud_talking'
}

TARGET_CLASSES = ['Silence', 'Whispering', 'Typing', 'Phone_ringing', 'Loud_talking']

class NoiseAnalyzer:
    def __init__(self, dataset_path='./ESC-50-master'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler_mean = None
        self.scaler_std = None
        
    def extract_features(self, audio, sr=SAMPLE_RATE):
        """Extract mel spectrogram features from audio"""
        # Ensure audio is exactly 1 second
        target_length = int(DURATION * sr)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        try:
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmax=sr//2
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to get (time, frequency) shape
            mel_spec_transposed = mel_spec_db.T
            
            # Debug info (only print occasionally to avoid spam)
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
            
            if self._debug_count % 20 == 1:  # Print every 20th extraction
                print(f"\nDebug - Audio length: {len(audio)}, Mel spec shape: {mel_spec.shape}, Transposed: {mel_spec_transposed.shape}")
            
            # Ensure we have exactly N_FRAMES by taking the first N_FRAMES or padding
            if mel_spec_transposed.shape[0] >= N_FRAMES:
                mel_spec_transposed = mel_spec_transposed[:N_FRAMES]  # Take first N_FRAMES
            else:
                # Pad with the last frame repeated if we have fewer frames
                padding_frames = N_FRAMES - mel_spec_transposed.shape[0]
                if mel_spec_transposed.shape[0] > 0:
                    last_frame = mel_spec_transposed[-1:, :]  # Shape: (1, N_MELS)
                    padding = np.repeat(last_frame, padding_frames, axis=0)
                else:
                    padding = np.zeros((padding_frames, N_MELS))
                mel_spec_transposed = np.vstack([mel_spec_transposed, padding])
            
            # Verify final shape
            assert mel_spec_transposed.shape == (N_FRAMES, N_MELS), f"Expected ({N_FRAMES}, {N_MELS}), got {mel_spec_transposed.shape}"
            
            # Add channel dimension and return
            return mel_spec_transposed.reshape(N_FRAMES, N_MELS, 1)
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return zeros as fallback
            return np.zeros((N_FRAMES, N_MELS, 1))
    
    def augment_audio(self, audio, sr=SAMPLE_RATE):
        """Apply data augmentation techniques"""
        augmented = []
        
        # Original
        augmented.append(audio)
        
        # Time stretching (0.8x to 1.2x)
        for rate in [0.9, 1.1]:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            if len(stretched) >= int(DURATION * sr * 0.8):  # Ensure minimum length
                augmented.append(stretched)
        
        # Pitch shifting (±2 semitones)
        for n_steps in [-1, 1]:
            pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            augmented.append(pitched)
        
        # Add background noise
        noise = np.random.normal(0, 0.005, len(audio))
        noisy = audio + noise
        augmented.append(noisy)
        
        # Time shifting
        shift = int(0.1 * len(audio))
        if shift > 0:
            shifted = np.roll(audio, shift)
            augmented.append(shifted)
        
        return augmented
    
    def load_esc50_dataset(self):
        """Load and prepare ESC-50 dataset"""
        print("Loading ESC-50 dataset...")
        
        # Load metadata
        meta_path = os.path.join(self.dataset_path, 'meta', 'esc50.csv')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"ESC-50 metadata not found at {meta_path}")
        
        meta_df = pd.read_csv(meta_path)
        
        # Filter for relevant categories
        relevant_files = meta_df[meta_df['category'].isin(ESC50_TO_TARGET.keys())]
        print(f"Found {len(relevant_files)} relevant audio files")
        
        # Check available categories
        available_categories = relevant_files['category'].unique()
        print(f"Available ESC-50 categories: {available_categories}")
        
        features = []
        labels = []
        
        audio_dir = os.path.join(self.dataset_path, 'audio')
        
        for idx, row in relevant_files.iterrows():
            try:
                # Load audio file
                audio_path = os.path.join(audio_dir, row['filename'])
                audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                
                # Get target label
                target_label = ESC50_TO_TARGET[row['category']]
                
                # Apply augmentation for training data
                augmented_audios = self.augment_audio(audio)
                
                for aug_audio in augmented_audios:
                    feature = self.extract_features(aug_audio)
                    features.append(feature)
                    labels.append(target_label)
                
                if len(features) % 50 == 0:
                    print(f"Processed {len(features)} samples...")
                    
            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
                continue
        
        # Check which target classes we have so far
        unique_labels = np.unique(labels)
        print(f"Target classes found in ESC-50: {unique_labels}")
        
        # Add synthetic samples for missing classes to ensure all 5 classes are present
        print("Generating synthetic samples for missing classes...")
        
        # Add synthetic silence samples
        for _ in range(150):
            silence = np.random.normal(0, 0.001, int(DURATION * SAMPLE_RATE))
            feature = self.extract_features(silence)
            features.append(feature)
            labels.append('Silence')
        
        # Add synthetic whispering samples (very quiet speech-like sounds)
        for _ in range(100):
            # Generate quiet periodic sounds resembling whispers
            t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
            whisper = np.random.normal(0, 0.02, len(t)) * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 2)
            feature = self.extract_features(whisper)
            features.append(feature)
            labels.append('Whispering')
        
        # Add synthetic typing samples if not enough
        if 'Typing' not in unique_labels:
            for _ in range(80):
                # Generate clicking sounds resembling typing
                typing_sound = np.zeros(int(DURATION * SAMPLE_RATE))
                # Add random clicks
                for _ in range(np.random.randint(5, 15)):
                    click_pos = np.random.randint(0, len(typing_sound) - 1000)
                    click = np.random.exponential(0.01, 1000) * np.random.choice([-1, 1])
                    typing_sound[click_pos:click_pos + 1000] += click
                feature = self.extract_features(typing_sound)
                features.append(feature)
                labels.append('Typing')
        
        # Add synthetic phone ringing if not enough
        if 'Phone_ringing' not in unique_labels:
            for _ in range(60):
                # Generate ringing sounds
                t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
                ring = 0.3 * np.sin(2 * np.pi * 800 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
                feature = self.extract_features(ring)
                features.append(feature)
                labels.append('Phone_ringing')
        
        # Add synthetic loud talking if not enough
        if 'Loud_talking' not in unique_labels or len([l for l in labels if l == 'Loud_talking']) < 50:
            for _ in range(120):
                # Generate loud speech-like sounds
                t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
                loud_talk = np.random.normal(0, 0.1, len(t)) * np.sin(2 * np.pi * 150 * t + np.random.random() * 2 * np.pi)
                loud_talk += np.random.normal(0, 0.05, len(t)) * np.sin(2 * np.pi * 300 * t + np.random.random() * 2 * np.pi)
                feature = self.extract_features(loud_talk)
                features.append(feature)
                labels.append('Loud_talking')
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Total samples: {len(features)}")
        print(f"Feature shape: {features.shape}")
        
        # Check final class distribution
        unique_labels_final, counts = np.unique(labels, return_counts=True)
        print("Final class distribution:")
        for label, count in zip(unique_labels_final, counts):
            print(f"  {label}: {count} samples")
        
        # Ensure we have all 5 target classes
        missing_classes = set(TARGET_CLASSES) - set(unique_labels_final)
        if missing_classes:
            raise ValueError(f"Missing target classes: {missing_classes}")
        
        # Encode labels with all target classes
        self.label_encoder.fit(TARGET_CLASSES)  # Fit with all classes first
        labels_encoded = self.label_encoder.transform(labels)
        labels_onehot = tf.keras.utils.to_categorical(labels_encoded, len(TARGET_CLASSES))
        
        # Normalize features
        features_reshaped = features.reshape(-1, features.shape[-1])
        self.scaler_mean = np.mean(features_reshaped, axis=0)
        self.scaler_std = np.std(features_reshaped, axis=0) + 1e-8
        
        features_normalized = (features_reshaped - self.scaler_mean) / self.scaler_std
        features_normalized = features_normalized.reshape(features.shape)
        
        return features_normalized, labels_onehot, labels
    
    def create_model(self):
        """Create the CNN-RNN model architecture"""
        model = Sequential([
            # First Conv1D block
            Conv1D(16, kernel_size=3, activation='relu', input_shape=(N_FRAMES, N_MELS)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # Second Conv1D block
            Conv1D(32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # Third Conv1D block
            Conv1D(64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # RNN layer
            GRU(32, return_sequences=False),
            Dropout(0.25),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(len(TARGET_CLASSES), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, features, labels, labels_raw):
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
        self.model = self.create_model()
        print(self.model.summary())
        
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
        
        history = self.model.fit(
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
        test_loss, test_accuracy = self.model.evaluate(X_test_reshaped, y_test)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Generate predictions
        y_pred = self.model.predict(X_test_reshaped)
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
            print(f"{class_name} F1-score: {f1_scores[i]:.4f}")
        
        # Plot training curves
        self.plot_training_curves(history)
        
        # Save model
        self.model.save('models/noise_classifier_model.h5')
        print("Model saved as 'models/noise_classifier_model.h5'")
        
        # Save normalization parameters
        np.savez('model_params.npz', 
                scaler_mean=self.scaler_mean, 
                scaler_std=self.scaler_std)
        print("Normalization parameters saved as 'model_params.npz'")
        
        return history
    
    def plot_training_curves(self, history):
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
    
    def load_model(self, model_path='models/best_model.h5'):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
        # Initialize label encoder with target classes for inference
        self.label_encoder.fit(TARGET_CLASSES)
        
        # Try to load saved normalization parameters
        params_path = 'model_params.npz'
        if os.path.exists(params_path):
            try:
                params = np.load(params_path)
                self.scaler_mean = params['scaler_mean']
                self.scaler_std = params['scaler_std']
                print(f"Loaded normalization parameters from {params_path}")
            except Exception as e:
                print(f"Warning: Could not load normalization parameters: {e}")
                print("Using default normalization parameters")
                self.scaler_mean = np.zeros(N_MELS)
                self.scaler_std = np.ones(N_MELS)
        else:
            print("No normalization parameters found, using defaults")
            self.scaler_mean = np.zeros(N_MELS)
            self.scaler_std = np.ones(N_MELS)
        
        print(f"Model loaded from {model_path}")
        print("Label encoder and normalization parameters initialized for inference")
        
        # Validate model architecture
        try:
            # Test the model with a dummy input
            dummy_input = np.random.random((1, N_FRAMES, N_MELS))
            dummy_output = self.model.predict(dummy_input, verbose=0)
            expected_classes = len(TARGET_CLASSES)
            
            print(f"Model validation:")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {dummy_output.shape}")
            print(f"  Output type: {type(dummy_output)}")
            print(f"  Expected classes: {expected_classes}")
            
            if len(dummy_output.shape) > 1:
                actual_classes = dummy_output.shape[-1]
                print(f"  Actual output classes: {actual_classes}")
                print(f"  Output sample: {dummy_output[0] if dummy_output.shape[0] > 0 else 'empty'}")
            else:
                actual_classes = len(dummy_output) if hasattr(dummy_output, '__len__') else 1
                print(f"  Actual output classes (1D): {actual_classes}")
                print(f"  Output sample: {dummy_output}")
            
            if actual_classes != expected_classes:
                print(f"WARNING: Model output ({actual_classes}) doesn't match expected classes ({expected_classes})")
            else:
                print("✅ Model validation passed!")
                
        except Exception as e:
            print(f"Model validation failed: {e}")
            print("This might cause issues during prediction")
    
    def predict_audio(self, audio, sr=SAMPLE_RATE):
        """Predict class for audio sample"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Debug: Check feature shape before normalization
        if hasattr(self, '_predict_debug_count'):
            self._predict_debug_count += 1
        else:
            self._predict_debug_count = 1
            
        if self._predict_debug_count % 10 == 1:
            print(f"Debug predict - Features shape: {features.shape}")
        
        # Normalize using training statistics
        if self.scaler_mean is not None and self.scaler_std is not None:
            # Debug normalization parameters
            if self._predict_debug_count % 10 == 1:
                print(f"Debug normalize - scaler_mean shape: {self.scaler_mean.shape if hasattr(self.scaler_mean, 'shape') else 'no shape'}")
                print(f"Debug normalize - scaler_std shape: {self.scaler_std.shape if hasattr(self.scaler_std, 'shape') else 'no shape'}")
                print(f"Debug normalize - N_MELS: {N_MELS}")
            
            # Check if normalization parameters have correct dimensions
            if (hasattr(self.scaler_mean, '__len__') and len(self.scaler_mean) != N_MELS) or \
               (hasattr(self.scaler_std, '__len__') and len(self.scaler_std) != N_MELS):
                print(f"Warning: Normalization parameter mismatch - mean: {len(self.scaler_mean) if hasattr(self.scaler_mean, '__len__') else 'scalar'}, std: {len(self.scaler_std) if hasattr(self.scaler_std, '__len__') else 'scalar'}, expected: {N_MELS}")
                print("Skipping normalization")
            else:
                try:
                    # Remove channel dimension for normalization, then add it back
                    features_2d = features.reshape(N_FRAMES, N_MELS)  # (100, 40)
                    
                    # Normalize each mel bin
                    for i in range(min(N_MELS, len(self.scaler_std) if hasattr(self.scaler_std, '__len__') else N_MELS)):
                        if len(self.scaler_std) > i and self.scaler_std[i] > 0:
                            features_2d[:, i] = (features_2d[:, i] - self.scaler_mean[i]) / self.scaler_std[i]
                    
                    # Add channel dimension back
                    features = features_2d.reshape(N_FRAMES, N_MELS, 1)
                    
                except Exception as norm_err:
                    print(f"Error in normalization: {norm_err}")
                    print("Continuing without normalization")
        
        # Reshape for prediction (remove channel dimension for Conv1D)
        features_reshaped = features.reshape(1, N_FRAMES, N_MELS)
        
        # Debug: Check final input shape
        if self._predict_debug_count % 10 == 1:
            print(f"Debug predict - Input to model shape: {features_reshaped.shape}")
            print(f"Debug predict - Model expects: (batch_size, {N_FRAMES}, {N_MELS})")
            print(f"Debug predict - Model output classes: {len(TARGET_CLASSES)}")
        
        # Predict
        try:
            prediction = self.model.predict(features_reshaped, verbose=0)
            
            # Debug prediction shape
            if self._predict_debug_count % 10 == 1:
                print(f"Debug predict - Prediction shape: {prediction.shape}, Prediction: {prediction}")
            
            # Ensure we have a valid prediction array
            if prediction is None or len(prediction) == 0:
                print("Warning: Empty prediction, using default")
                return TARGET_CLASSES[0], 0.2, np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            
            # Get the prediction for the single sample
            pred_probs = prediction[0] if len(prediction.shape) > 1 and prediction.shape[0] > 0 else prediction
            
            # Debug prediction extraction
            if self._predict_debug_count % 10 == 1:
                print(f"Debug predict - After extraction, pred_probs shape: {pred_probs.shape if hasattr(pred_probs, 'shape') else 'no shape'}, len: {len(pred_probs) if hasattr(pred_probs, '__len__') else 'no len'}")
                print(f"Debug predict - pred_probs content: {pred_probs}")
            
            # Convert to numpy array if needed
            if not isinstance(pred_probs, np.ndarray):
                pred_probs = np.array(pred_probs)
            
            # Flatten if needed
            pred_probs = pred_probs.flatten()
            
            # Ensure we have the right number of classes
            if len(pred_probs) != len(TARGET_CLASSES):
                if self._predict_debug_count % 10 == 1:
                    print(f"Warning: Prediction has {len(pred_probs)} classes, expected {len(TARGET_CLASSES)}")
                # Pad or truncate as needed
                if len(pred_probs) < len(TARGET_CLASSES):
                    pred_probs = np.pad(pred_probs, (0, len(TARGET_CLASSES) - len(pred_probs)), constant_values=0.0)
                else:
                    pred_probs = pred_probs[:len(TARGET_CLASSES)]
            
            # Ensure pred_probs has valid values
            if len(pred_probs) == 0:
                print("Warning: Empty prediction probabilities, using defaults")
                pred_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            
            # Safe argmax with bounds checking
            try:
                predicted_class_idx = np.argmax(pred_probs)
                if predicted_class_idx >= len(pred_probs) or predicted_class_idx < 0:
                    predicted_class_idx = 0
                confidence = float(pred_probs[predicted_class_idx]) if len(pred_probs) > predicted_class_idx else 0.2
            except (IndexError, ValueError) as idx_err:
                print(f"Warning: Index error in prediction: {idx_err}")
                predicted_class_idx = 0
                confidence = 0.2
            
            # Ensure the predicted class index is valid
            if predicted_class_idx < len(TARGET_CLASSES):
                predicted_class = TARGET_CLASSES[predicted_class_idx]
            else:
                predicted_class = TARGET_CLASSES[0]  # Default to first class if index is invalid
                print(f"Warning: Invalid class index {predicted_class_idx}, using default class")
            
            return predicted_class, float(confidence), pred_probs
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return safe defaults
            return TARGET_CLASSES[0], 0.2, np.array([0.2, 0.2, 0.2, 0.2, 0.2])

class RealTimeClassifier:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.is_running = False
        self.predictions_log = []
        self.audio_buffer = []
        self.buffer_size = int(SAMPLE_RATE * DURATION)
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for real-time audio capture"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add new audio to buffer
        self.audio_buffer.extend(indata[:, 0])  # Take only first channel
        
        # Keep buffer size manageable
        if len(self.audio_buffer) > self.buffer_size * 2:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
    
    def audio_callback_adaptive(self, indata, frames, time, status):
        """Adaptive callback for different sample rates"""
        # Only print status if it's a serious error (not overflow/underflow which are common)
        if status and 'overflow' not in str(status).lower() and 'underflow' not in str(status).lower():
            print(f"Audio callback status: {status}")
        
        try:
            # Add new audio to buffer
            if indata is not None and len(indata.shape) >= 2:
                self.audio_buffer.extend(indata[:, 0])  # Take only first channel
            elif indata is not None:
                self.audio_buffer.extend(indata)  # Already single channel
            
            # Keep buffer size manageable - be more aggressive with trimming
            max_buffer_size = self.buffer_size * 3
            if len(self.audio_buffer) > max_buffer_size:
                self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        except Exception as e:
            print(f"Error in audio callback: {e}")
    
    def start_real_time_prediction(self):
        """Start real-time prediction with microphone input"""
        print("Starting real-time noise classification...")
        
        # List available audio devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (max channels: {device['max_input_channels']})")
                input_devices.append(i)
        
        if not input_devices:
            print("No input devices found!")
            return
        
        # Let user choose device or use default
        try:
            device_choice = input(f"\nEnter device number (default: {sd.default.device[0]}): ").strip()
            if device_choice:
                device_id = int(device_choice)
            else:
                device_id = sd.default.device[0]
        except (ValueError, IndexError):
            device_id = sd.default.device[0]
            
        selected_device = devices[device_id]
        print(f"Using device: {selected_device['name']}")
        
        # Try to find a compatible sample rate
        supported_rates = [SAMPLE_RATE, 44100, 48000, 22050, 8000]
        working_rate = None
        
        for rate in supported_rates:
            try:
                # Test if this sample rate works
                sd.check_input_settings(device=device_id, channels=1, samplerate=rate)
                working_rate = rate
                print(f"Using sample rate: {rate} Hz")
                break
            except Exception as e:
                print(f"Sample rate {rate} Hz not supported: {e}")
                continue
        
        if working_rate is None:
            print("No compatible sample rate found for this device!")
            return
            
        # Update global sample rate if different
        actual_sample_rate = working_rate
        actual_hop_length = int(HOP_LENGTH * actual_sample_rate / SAMPLE_RATE)
        # Use larger buffer to prevent overflow
        actual_blocksize = actual_hop_length * 4
        
        # Test microphone for 3 seconds
        print(f"\nTesting microphone for 3 seconds at {actual_sample_rate} Hz...")
        print("Please make some noise to verify audio input is working...")
        
        test_audio = []
        def test_callback(indata, frames, time, status):
            test_audio.extend(indata[:, 0])
        
        try:
            with sd.InputStream(callback=test_callback, device=device_id,
                              channels=1, samplerate=actual_sample_rate, blocksize=actual_blocksize):
                for i in range(3):
                    time.sleep(1)
                    if len(test_audio) > actual_sample_rate:
                        recent_audio = np.array(test_audio[-actual_sample_rate:])
                        rms = np.sqrt(np.mean(recent_audio**2))
                        max_val = np.max(np.abs(recent_audio))
                        level_bar_length = min(int(rms * 1000), 40)
                        level_bar = "█" * level_bar_length + "░" * (40 - level_bar_length)
                        print(f"Audio Level {i+1}/3: |{level_bar}| RMS={rms:.4f}")
                        
            if len(test_audio) == 0:
                print("WARNING: No audio detected! Check your microphone.")
                return
            else:
                final_rms = np.sqrt(np.mean(np.array(test_audio)**2))
                if final_rms < 0.001:
                    print("WARNING: Very low audio levels detected. Check microphone volume.")
                else:
                    print("✅ Microphone test passed!")
                    
        except Exception as e:
            print(f"Microphone test failed: {e}")
            return
        
        print("\nPress Ctrl+C to stop")
        print("Predictions will be shown every second with detailed probabilities")
        print("-" * 80)
        
        self.is_running = True
        prediction_count = 0
        
        # Store audio parameters for the callback
        self.actual_sample_rate = actual_sample_rate
        self.buffer_size = int(actual_sample_rate * DURATION)
        
        # Start audio stream with specified device and working sample rate
        try:
            with sd.InputStream(callback=self.audio_callback_adaptive, 
                              device=device_id,
                              channels=1, samplerate=actual_sample_rate,
                              blocksize=actual_blocksize):
                
                while self.is_running:
                    if len(self.audio_buffer) >= self.buffer_size:
                        # Get the latest 1-second window
                        audio_window = np.array(self.audio_buffer[-self.buffer_size:])
                        
                        try:
                            # Validate audio window
                            if len(audio_window) == 0:
                                continue
                                
                            # Resample to 16kHz if needed
                            if hasattr(self, 'actual_sample_rate') and self.actual_sample_rate != SAMPLE_RATE:
                                # Resample audio to the expected 16kHz
                                audio_resampled = librosa.resample(audio_window, 
                                                                 orig_sr=self.actual_sample_rate, 
                                                                 target_sr=SAMPLE_RATE)
                            else:
                                audio_resampled = audio_window
                            
                            # Ensure exactly 1 second at 16kHz
                            target_samples = int(DURATION * SAMPLE_RATE)
                            if len(audio_resampled) > target_samples:
                                audio_resampled = audio_resampled[:target_samples]
                            elif len(audio_resampled) < target_samples:
                                audio_resampled = np.pad(audio_resampled, (0, target_samples - len(audio_resampled)))
                            
                            # Check for invalid audio data
                            if np.all(audio_resampled == 0) or np.all(np.isnan(audio_resampled)) or np.all(np.isinf(audio_resampled)):
                                print("Warning: Invalid audio data detected, skipping...")
                                continue
                            
                            # Check audio levels to verify input
                            audio_rms = np.sqrt(np.mean(audio_resampled**2))
                            audio_max = np.max(np.abs(audio_resampled))
                            
                            # Make prediction
                            predicted_class, confidence, all_probs = self.analyzer.predict_audio(audio_resampled)
                            
                            # Validate prediction results
                            if not isinstance(predicted_class, str) or confidence < 0 or confidence > 1:
                                print(f"Warning: Invalid prediction results: {predicted_class}, {confidence}")
                                continue
                            
                            # Debug all_probs shape
                            if prediction_count % 20 == 1:  # Debug every 20th prediction
                                print(f"Debug all_probs - Type: {type(all_probs)}, Shape: {all_probs.shape if hasattr(all_probs, 'shape') else 'no shape'}, Length: {len(all_probs) if hasattr(all_probs, '__len__') else 'no len'}")
                                print(f"Debug all_probs - Content: {all_probs}")
                            
                            # Ensure all_probs is a proper numpy array with correct length
                            if not isinstance(all_probs, np.ndarray):
                                all_probs = np.array(all_probs)
                            
                            if len(all_probs) != len(TARGET_CLASSES):
                                print(f"Warning: all_probs length {len(all_probs)} != TARGET_CLASSES length {len(TARGET_CLASSES)}")
                                # Fix the length
                                if len(all_probs) < len(TARGET_CLASSES):
                                    all_probs = np.pad(all_probs, (0, len(TARGET_CLASSES) - len(all_probs)), constant_values=0.0)
                                else:
                                    all_probs = all_probs[:len(TARGET_CLASSES)]
                            
                            # Log prediction with audio levels
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            self.predictions_log.append({
                                'timestamp': timestamp,
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'all_probabilities': all_probs.tolist(),
                                'audio_rms': float(audio_rms),
                                'audio_max': float(audio_max)
                            })
                            
                            prediction_count += 1
                            
                            # Display prediction every second (every 2 predictions due to 0.5s sleep)
                            if prediction_count % 2 == 0:
                                print(f"\n[{timestamp}] Second #{prediction_count//2}")
                                print(f"🎯 PREDICTION: {predicted_class} (Confidence: {confidence:.3f})")
                                print(f"🎤 Audio Levels: RMS={audio_rms:.4f}, Max={audio_max:.4f}")
                                
                                # Audio level indicator
                                level_bar_length = min(int(audio_rms * 1000), 40)  # Scale RMS for display
                                level_bar = "█" * level_bar_length + "░" * (40 - level_bar_length)
                                print(f"� Input Level: |{level_bar}|")
                                
                                print(" All Class Probabilities:")
                                
                                # Sort probabilities for better display - ensure all_probs is the right shape
                                if len(all_probs) >= len(TARGET_CLASSES):
                                    prob_pairs = [(TARGET_CLASSES[i], all_probs[i]) for i in range(len(TARGET_CLASSES))]
                                else:
                                    print(f"Warning: all_probs has shape {all_probs.shape}, expected at least {len(TARGET_CLASSES)} elements")
                                    # Pad with zeros if needed
                                    padded_probs = np.pad(all_probs, (0, max(0, len(TARGET_CLASSES) - len(all_probs))), constant_values=0.0)
                                    prob_pairs = [(TARGET_CLASSES[i], padded_probs[i]) for i in range(len(TARGET_CLASSES))]
                                
                                prob_pairs.sort(key=lambda x: x[1], reverse=True)
                                
                                for class_name, prob in prob_pairs:
                                    bar_length = int(prob * 30)  # Scale to 30 characters
                                    bar = "█" * bar_length + "░" * (30 - bar_length)
                                    print(f"  {class_name:<15}: {prob:.3f} |{bar}|")
                                
                                print("-" * 80)
                            else:
                                # Show quick update with audio level
                                level_indicator = "●" if audio_rms > 0.01 else "○"
                                print(f"\r[{timestamp}] {predicted_class:<15} ({confidence:.3f}) {level_indicator}", end='', flush=True)
                        
                        except Exception as e:
                            print(f"\nPrediction error: {e}")
                    
                    time.sleep(0.5)  # 50% overlap (500ms stride)
                    
        except KeyboardInterrupt:
            print("\nStopping real-time prediction...")
            self.stop_real_time_prediction()
        except Exception as e:
            print(f"Error in real-time prediction: {e}")
    
    def stop_real_time_prediction(self):
        """Stop real-time prediction and save logs"""
        self.is_running = False
        
        if self.predictions_log:
            # Save predictions to CSV
            df = pd.DataFrame(self.predictions_log)
            df.to_csv('real_time_predictions.csv', index=False)
            print(f"\nSaved {len(self.predictions_log)} predictions to 'real_time_predictions.csv'")
            
            # Display detailed summary
            print("\n" + "="*60)
            print("PREDICTION SUMMARY")
            print("="*60)
            
            # Class distribution
            class_counts = df['predicted_class'].value_counts()
            total_predictions = len(df)
            
            print(f"Total predictions: {total_predictions}")
            print(f"Session duration: ~{total_predictions * 0.5:.1f} seconds")
            print("\nClass Distribution:")
            
            for class_name in TARGET_CLASSES:
                count = class_counts.get(class_name, 0)
                percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
                bar_length = int(percentage / 100 * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"  {class_name:<15}: {count:3d} ({percentage:5.1f}%) |{bar}|")
            
            # Average confidence per class
            print("\nAverage Confidence by Class:")
            for class_name in TARGET_CLASSES:
                class_data = df[df['predicted_class'] == class_name]
                if len(class_data) > 0:
                    avg_conf = class_data['confidence'].mean()
                    print(f"  {class_name:<15}: {avg_conf:.3f}")
                else:
                    print(f"  {class_name:<15}: No predictions")
            
            print("="*60)

def main():
    print("5-Class Audio Classification System for Library Noise Monitoring")
    print("="*65)
    
    # Initialize analyzer
    analyzer = NoiseAnalyzer()
    
    choice = input("Choose mode:\n1. Train new model\n2. Load existing model for real-time prediction\nEnter choice (1 or 2): ")
    
    if choice == '1':
        print("\nTraining mode selected")
        
        # Check if ESC-50 dataset exists
        if not os.path.exists(analyzer.dataset_path):
            print(f"ESC-50 dataset not found at {analyzer.dataset_path}")
            print("Please download ESC-50 dataset from: https://github.com/karolpiczak/ESC-50")
            print("Extract it to the current directory as 'ESC-50-master'")
            return
        
        try:
            # Load and prepare dataset
            features, labels, labels_raw = analyzer.load_esc50_dataset()
            
            # Train model
            history = analyzer.train_model(features, labels, labels_raw)
            
            # Ask if user wants to test real-time prediction
            test_realtime = input("\nWould you like to test real-time prediction? (y/n): ")
            if test_realtime.lower() == 'y':
                rt_classifier = RealTimeClassifier(analyzer)
                rt_classifier.start_real_time_prediction()
            
        except Exception as e:
            print(f"Error during training: {e}")
            return
    
    elif choice == '2':
        print("\nReal-time prediction mode selected")
        
        # Load pre-trained model
        model_path = input("Enter model path (default: models/best_model.h5): ").strip()
        if not model_path:
            model_path = 'models/best_model.h5'
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
        
        try:
            analyzer.load_model(model_path)
            
            # Initialize real-time classifier
            rt_classifier = RealTimeClassifier(analyzer)
            rt_classifier.start_real_time_prediction()
            
        except Exception as e:
            print(f"Error in real-time prediction: {e}")
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()