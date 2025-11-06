import numpy as np
import librosa
import time
from collections import deque

class VoiceActivityDetector:
    """
    Voice Activity Detection system specifically designed for library environments.
    Uses calibration-based threshold with multiple acoustic features to avoid
    overfitting to continuous ambient noise.
    """
    
    def __init__(self, sample_rate=16000, frame_length=400, hop_length=160):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Calibration parameters
        self.calibration_duration = 3.5  # seconds
        self.is_calibrated = False
        self.calibration_samples = []
        
        # Thresholds (will be set during calibration)
        self.energy_threshold = None
        self.spectral_centroid_threshold = None
        self.zero_crossing_threshold = None
        self.mfcc_variance_threshold = None
        
        # Safety margins to prevent false positives (made more sensitive for testing)
        self.energy_margin = 1.3      # 1.3x above calibrated silence (was 2.0)
        self.spectral_margin = 1.2    # 1.2x above calibrated silence (was 1.5)
        self.zcr_margin = 1.3         # 1.3x above calibrated silence (was 1.8)
        self.mfcc_margin = 1.4        # 1.4x above calibrated silence (was 2.2)
        
        # Minimum duration for activity (to filter out brief spikes)
        self.min_activity_duration = 0.3  # seconds
        self.activity_buffer = deque(maxlen=10)  # Store last 10 decisions
        
        print("VAD initialized. Calibration needed for 3.5 seconds.")
    
    def extract_features(self, audio):
        """Extract multiple acoustic features for robust VAD"""
        if len(audio) == 0:
            return None, None, None, None
            
        # 1. Energy (RMS)
        energy = np.sqrt(np.mean(audio**2))
        
        # 2. Spectral Centroid (brightness of sound)
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0])
        except:
            spectral_centroid = 0
        
        # 3. Zero Crossing Rate (measure of noisiness)
        zcr = np.mean(librosa.feature.zero_crossing_rate(
            audio, frame_length=self.frame_length, hop_length=self.hop_length)[0])
        
        # 4. MFCC variance (speech characteristics)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, 
                                       hop_length=self.hop_length)
            mfcc_variance = np.mean(np.var(mfccs, axis=1))
        except:
            mfcc_variance = 0
        
        return energy, spectral_centroid, zcr, mfcc_variance
    
    def add_calibration_sample(self, audio):
        """Add audio sample to calibration data"""
        if self.is_calibrated:
            return True
            
        features = self.extract_features(audio)
        if any(f is None for f in features):
            return False
            
        self.calibration_samples.append(features)
        
        # Check if we have enough calibration data
        total_duration = len(self.calibration_samples) * len(audio) / self.sample_rate
        
        if total_duration >= self.calibration_duration:
            self._compute_thresholds()
            self.is_calibrated = True
            print(f"VAD calibrated after {total_duration:.1f} seconds")
            print(f"Thresholds - Energy: {self.energy_threshold:.6f}, "
                  f"Spectral: {self.spectral_centroid_threshold:.1f}, "
                  f"ZCR: {self.zero_crossing_threshold:.4f}, "
                  f"MFCC Var: {self.mfcc_variance_threshold:.4f}")
            return True
            
        return False
    
    def _compute_thresholds(self):
        """Compute thresholds based on calibration samples"""
        if not self.calibration_samples:
            raise ValueError("No calibration samples available")
        
        # Convert to numpy array for easier processing
        features_array = np.array(self.calibration_samples)
        
        # Calculate robust statistics (use 95th percentile instead of max to handle outliers)
        energy_baseline = np.percentile(features_array[:, 0], 95)
        spectral_baseline = np.percentile(features_array[:, 1], 95)
        zcr_baseline = np.percentile(features_array[:, 2], 95)
        mfcc_baseline = np.percentile(features_array[:, 3], 95)
        
        # Set thresholds with safety margins
        self.energy_threshold = energy_baseline * self.energy_margin
        self.spectral_centroid_threshold = spectral_baseline * self.spectral_margin
        self.zero_crossing_threshold = zcr_baseline * self.zcr_margin
        self.mfcc_variance_threshold = mfcc_baseline * self.mfcc_margin
        
        # Ensure minimum thresholds to prevent overly sensitive detection (made more sensitive)
        self.energy_threshold = max(self.energy_threshold, 0.002)  # Minimum energy (was 0.005)
        self.spectral_centroid_threshold = max(self.spectral_centroid_threshold, 800)  # Min freq (was 1000)
        self.zero_crossing_threshold = max(self.zero_crossing_threshold, 0.005)  # Min ZCR (was 0.01)
        self.mfcc_variance_threshold = max(self.mfcc_variance_threshold, 0.05)  # Min MFCC var (was 0.1)
    
    def detect_activity(self, audio):
        """
        Detect voice activity in audio sample
        Returns: (is_activity, confidence, debug_info)
        """
        if not self.is_calibrated:
            return False, 0.0, "Not calibrated"
        
        features = self.extract_features(audio)
        if any(f is None for f in features):
            return False, 0.0, "Feature extraction failed"
        
        energy, spectral_centroid, zcr, mfcc_variance = features
        
        # Individual feature decisions
        energy_active = energy > self.energy_threshold
        spectral_active = spectral_centroid > self.spectral_centroid_threshold
        zcr_active = zcr > self.zero_crossing_threshold
        mfcc_active = mfcc_variance > self.mfcc_variance_threshold
        
        # Weighted decision (energy is most important for library environment)
        feature_scores = [
            energy_active * 0.4,      # 40% weight - primary indicator
            spectral_active * 0.25,   # 25% weight - frequency content
            zcr_active * 0.2,         # 20% weight - noisiness
            mfcc_active * 0.15        # 15% weight - speech characteristics
        ]
        
        confidence = sum(feature_scores)
        is_activity = confidence > 0.5  # Majority vote with weights
        
        # Temporal smoothing to prevent brief false positives
        self.activity_buffer.append(is_activity)
        
        # Require sustained activity for positive detection (made less restrictive)
        if is_activity:
            recent_activity = sum(self.activity_buffer) / len(self.activity_buffer)
            # Need at least 40% of recent samples to be active (was 60%)
            is_activity = recent_activity >= 0.4
            confidence = recent_activity
        
        debug_info = {
            'energy': energy,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr,
            'mfcc_variance': mfcc_variance,
            'energy_active': energy_active,
            'spectral_active': spectral_active,
            'zcr_active': zcr_active,
            'mfcc_active': mfcc_active,
            'raw_confidence': sum(feature_scores),
            'smoothed_confidence': confidence,
            'buffer_activity': recent_activity if is_activity else sum(self.activity_buffer) / len(self.activity_buffer)
        }
        
        return is_activity, confidence, debug_info
    
    def reset_calibration(self):
        """Reset calibration to recalibrate for new environment"""
        self.is_calibrated = False
        self.calibration_samples = []
        self.activity_buffer.clear()
        print("VAD calibration reset. Need 3.5 seconds of silence for recalibration.")
    
    def get_calibration_progress(self):
        """Get calibration progress as percentage"""
        if self.is_calibrated:
            return 100.0
        
        current_duration = len(self.calibration_samples) * 1.0  # Assuming 1-second samples
        return min(100.0, (current_duration / self.calibration_duration) * 100)
    
    def is_ready(self):
        """Check if VAD is ready for use"""
        return self.is_calibrated
