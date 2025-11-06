import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
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
LEARNING_RATE = 0.001

# Class mapping from ESC-50 to target classes (excluding silence - handled by VAD)
ESC50_TO_TARGET = {
    # Whispering - closest approximations
    'sneezing': 'Whispering',
    'breathing': 'Whispering',
    'wind': 'Whispering',
    
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

TARGET_CLASSES = ['Whispering', 'Typing', 'Phone_ringing', 'Loud_talking']

class NoiseAnalyzer:
    def __init__(self, dataset_path='../data/ESC-50-master'):
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
            
            # if self._debug_count % 20 == 1:  # Print every 20th extraction
                # print(f"\nDebug - Audio length: {len(audio)}, Mel spec shape: {mel_spec.shape}, Transposed: {mel_spec_transposed.shape}")
            
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
        
        # Add synthetic samples for missing classes to ensure all 4 classes are present
        print("Generating synthetic samples for missing classes...")
        
        # Add synthetic whispering samples (very quiet speech-like sounds)
        if 'Whispering' not in unique_labels or len([l for l in labels if l == 'Whispering']) < 50:
            for _ in range(120):
                # Generate quiet periodic sounds resembling whispers
                t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
                whisper = np.random.normal(0, 0.02, len(t)) * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 2)
                # Add some randomness to make it more realistic
                whisper += np.random.normal(0, 0.005, len(t))
                feature = self.extract_features(whisper)
                features.append(feature)
                labels.append('Whispering')
        
        # Add synthetic typing samples if not enough
        if 'Typing' not in unique_labels or len([l for l in labels if l == 'Typing']) < 50:
            for _ in range(100):
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
        if 'Phone_ringing' not in unique_labels or len([l for l in labels if l == 'Phone_ringing']) < 50:
            for _ in range(80):
                # Generate ringing sounds
                t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
                ring = 0.3 * np.sin(2 * np.pi * 800 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
                # Add some harmonics
                ring += 0.1 * np.sin(2 * np.pi * 1600 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
                feature = self.extract_features(ring)
                features.append(feature)
                labels.append('Phone_ringing')
        
        # Add synthetic loud talking if not enough
        if 'Loud_talking' not in unique_labels or len([l for l in labels if l == 'Loud_talking']) < 50:
            for _ in range(150):
                # Generate loud speech-like sounds
                t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
                loud_talk = np.random.normal(0, 0.1, len(t)) * np.sin(2 * np.pi * 150 * t + np.random.random() * 2 * np.pi)
                loud_talk += np.random.normal(0, 0.05, len(t)) * np.sin(2 * np.pi * 300 * t + np.random.random() * 2 * np.pi)
                # Add some burst-like characteristics
                loud_talk *= np.random.exponential(1, len(t))
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
        
        # Ensure we have all 4 target classes
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
    
    def load_model(self, model_path='models/best_model.h5'):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
        # Initialize label encoder with target classes for inference
        self.label_encoder.fit(TARGET_CLASSES)
        
        # Try to load saved normalization parameters
        params_path = 'models/model_params.npz'
        if os.path.exists(params_path):
            params = np.load(params_path)
            self.scaler_mean = params['scaler_mean']
            self.scaler_std = params['scaler_std']
            print(f"Loaded normalization parameters from {params_path}")
            print(f"Scaler mean shape: {self.scaler_mean.shape}")
            print(f"Scaler std shape: {self.scaler_std.shape}")
        else:
            print(f"Warning: Normalization parameters not found at {params_path}")
            print("Using default normalization (mean=0, std=1)")
            self.scaler_mean = None
            self.scaler_std = None
        
        print(f"Model loaded from {model_path}")
        print("Label encoder and normalization parameters initialized for inference")
        
        # Validate model architecture
        try:
            expected_input_shape = (None, N_FRAMES, N_MELS)
            actual_input_shape = self.model.input_shape
            print(f"Model input shape: {actual_input_shape}")
            print(f"Expected input shape: {expected_input_shape}")
            
            if actual_input_shape != expected_input_shape:
                print(f"Warning: Model input shape mismatch!")
                print(f"  Expected: {expected_input_shape}")
                print(f"  Actual: {actual_input_shape}")
                
                # Try to handle common shape mismatches
                if len(actual_input_shape) == 3 and actual_input_shape[1:] == (N_FRAMES, N_MELS):
                    print("Shape looks compatible, continuing...")
                elif len(actual_input_shape) == 4 and actual_input_shape[1:3] == (N_FRAMES, N_MELS):
                    print("Model expects 4D input (with channel dimension), will adapt...")
                else:
                    print("Warning: Significant shape mismatch detected!")
            
            expected_output_shape = (None, len(TARGET_CLASSES))
            actual_output_shape = self.model.output_shape
            print(f"Model output shape: {actual_output_shape}")
            print(f"Expected output shape: {expected_output_shape}")
            
            if actual_output_shape != expected_output_shape:
                print(f"Warning: Model output shape mismatch!")
                print(f"  Expected: {expected_output_shape}")
                print(f"  Actual: {actual_output_shape}")
                
        except Exception as e:
            print(f"Error validating model architecture: {e}")
    
    def predict_audio(self, audio, sr=SAMPLE_RATE):
        """Predict class for audio sample"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Debug: Check feature shape before normalization
        if hasattr(self, '_predict_debug_count'):
            self._predict_debug_count += 1
        else:
            self._predict_debug_count = 1
            
        if self._predict_debug_count % 10 == 1:
            print(f"Debug predict_audio - Features shape before normalization: {features.shape}")
        
        # Normalize using training statistics
        if self.scaler_mean is not None and self.scaler_std is not None:
            features_reshaped = features.reshape(-1, features.shape[-1])
            if self._predict_debug_count % 10 == 1:
                print(f"Debug predict_audio - Features reshaped: {features_reshaped.shape}")
                print(f"Debug predict_audio - Scaler mean shape: {self.scaler_mean.shape}")
                print(f"Debug predict_audio - Scaler std shape: {self.scaler_std.shape}")
            
            # Check if shapes are compatible
            if features_reshaped.shape[1] != self.scaler_mean.shape[0]:
                print(f"Warning: Feature dimension mismatch!")
                print(f"  Feature channels: {features_reshaped.shape[1]}")
                print(f"  Scaler channels: {self.scaler_mean.shape[0]}")
                
                # Try to handle the mismatch
                if features_reshaped.shape[1] == 1 and self.scaler_mean.shape[0] == N_MELS:
                    # Features seem to have an extra channel dimension, remove it
                    features = features.reshape(N_FRAMES, N_MELS)
                    features_reshaped = features.reshape(-1)
                    print(f"Adjusted features shape: {features_reshaped.shape}")
                elif features_reshaped.shape[1] == N_MELS and self.scaler_mean.shape[0] == 1:
                    # Scaler seems to be for channel dimension only
                    print("Scaler appears to be for channel dimension, skipping normalization")
                    features_normalized = features_reshaped.reshape(features.shape)
                else:
                    print("Cannot resolve shape mismatch, skipping normalization")
                    features_normalized = features
            else:
                # Normal normalization
                features_normalized_reshaped = (features_reshaped - self.scaler_mean) / self.scaler_std
                features_normalized = features_normalized_reshaped.reshape(features.shape)
        else:
            print("No normalization parameters available, using raw features")
            features_normalized = features
        
        # Reshape for prediction (remove channel dimension for Conv1D)
        features_reshaped = features.reshape(1, N_FRAMES, N_MELS)
        
        # Debug: Check final input shape
        if self._predict_debug_count % 10 == 1:
            print(f"Debug predict_audio - Final input shape: {features_reshaped.shape}")
            print(f"Debug predict_audio - Model input shape: {self.model.input_shape}")
        
        # Predict
        try:
            # Check if model expects 4D input (with channel dimension)
            if len(self.model.input_shape) == 4:
                # Model expects (batch, time, freq, channels)
                features_for_prediction = features_normalized.reshape(1, N_FRAMES, N_MELS, 1)
            else:
                # Model expects (batch, time, freq)
                features_for_prediction = features_normalized.reshape(1, N_FRAMES, N_MELS)
            
            if self._predict_debug_count % 10 == 1:
                print(f"Debug predict_audio - Input for prediction shape: {features_for_prediction.shape}")
            
            pred_probs = self.model.predict(features_for_prediction, verbose=0)
            
            if self._predict_debug_count % 10 == 1:
                print(f"Debug predict_audio - Raw prediction shape: {pred_probs.shape}")
                print(f"Debug predict_audio - Raw prediction: {pred_probs}")
            
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