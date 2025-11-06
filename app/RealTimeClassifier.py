import numpy as np
import pandas as pd
import sounddevice as sd
from app.NoiseAnalyzer import TARGET_CLASSES, SAMPLE_RATE, DURATION, HOP_LENGTH
from app.VAD import VoiceActivityDetector
import time
import librosa

class RealTimeClassifier:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
        self.is_running = False
        self.predictions_log = []
        self.audio_buffer = []
        self.buffer_size = int(SAMPLE_RATE * DURATION)
        self.calibration_phase = True
        
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
        
        print("\n" + "="*80)
        print("VOICE ACTIVITY DETECTION + NOISE CLASSIFICATION")
        print("="*80)
        print("Phase 1: VAD Calibration (3.5 seconds of silence needed)")
        print("Phase 2: Real-time classification with VAD pre-filtering")
        print("Press Ctrl+C to stop")
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
                            
                            # Check audio levels
                            audio_rms = np.sqrt(np.mean(audio_resampled**2))
                            audio_max = np.max(np.abs(audio_resampled))
                            
                            # PHASE 1: VAD Calibration
                            if not self.vad.is_ready():
                                calibration_complete = self.vad.add_calibration_sample(audio_resampled)
                                progress = self.vad.get_calibration_progress()
                                
                                # Show calibration progress
                                bar_length = int(progress / 100 * 40)
                                progress_bar = "█" * bar_length + "░" * (40 - bar_length)
                                print(f"\r🎙️ VAD Calibration: |{progress_bar}| {progress:.1f}% - Keep quiet!", end='', flush=True)
                                
                                if calibration_complete:
                                    print(f"\n✅ VAD Calibration complete! Now starting noise classification...")
                                    print("-" * 80)
                                continue
                            
                            # PHASE 2: VAD Detection + Noise Classification
                            is_activity, vad_confidence, vad_debug = self.vad.detect_activity(audio_resampled)
                            
                            # Debug VAD occasionally to help with troubleshooting
                            # if prediction_count % 10 == 1 and isinstance(vad_debug, dict):
                            #     print(f"\n🔍 VAD Debug - Energy: {vad_debug['energy']:.4f} (thresh: {self.vad.energy_threshold:.4f}), "
                            #           f"Spectral: {vad_debug['spectral_centroid']:.1f}Hz (thresh: {self.vad.spectral_centroid_threshold:.1f}), "
                            #           f"Activity: {is_activity}, Confidence: {vad_confidence:.3f}")
                            
                            if is_activity:
                                # Voice activity detected - use NoiseAnalyzer for classification
                                predicted_class, confidence, all_probs = self.analyzer.predict_audio(audio_resampled)
                                
                                # Validate prediction results
                                if not isinstance(predicted_class, str) or confidence < 0 or confidence > 1:
                                    print(f"Warning: Invalid prediction results: {predicted_class}, {confidence}")
                                    continue
                            else:
                                # No voice activity - classify as silence
                                predicted_class = "Silence"
                                confidence = vad_confidence
                                all_probs = np.zeros(len(TARGET_CLASSES))  # All zeros for non-target classes
                            
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
                            
                            # Log prediction with VAD info
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Create extended target classes including Silence for logging
                            extended_classes = ['Silence'] + TARGET_CLASSES
                            extended_probs = np.zeros(len(extended_classes))
                            
                            if predicted_class == "Silence":
                                extended_probs[0] = confidence  # Silence confidence
                            else:
                                extended_probs[0] = 1 - vad_confidence  # Silence probability from VAD
                                # Map other probabilities
                                for i, target_class in enumerate(TARGET_CLASSES):
                                    if i < len(all_probs):
                                        extended_probs[i + 1] = all_probs[i]
                            
                            self.predictions_log.append({
                                'timestamp': timestamp,
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'vad_activity': is_activity,
                                'vad_confidence': vad_confidence,
                                'all_probabilities': extended_probs.tolist(),
                                'audio_rms': float(audio_rms),
                                'audio_max': float(audio_max),
                                'vad_debug': vad_debug if is_activity else None
                            })
                            
                            prediction_count += 1
                            
                            # Display prediction every second (every 2 predictions due to 0.5s sleep)
                            if prediction_count % 2 == 0:
                                print(f"\n[{timestamp}] Second #{prediction_count//2}")
                                # VAD status indicator
                                vad_icon = "🔊" if is_activity else "🔇"
                                print(f"{vad_icon} VAD: {'Activity' if is_activity else 'Silence'} (Confidence: {vad_confidence:.3f})")
                                print(f"🎯 CLASSIFICATION: {predicted_class} (Confidence: {confidence:.3f})")
                                print(f"🎤 Audio Levels: RMS={audio_rms:.4f}, Max={audio_max:.4f}")
                                
                                # Audio level indicator
                                level_bar_length = min(int(audio_rms * 1000), 40)
                                level_bar = "█" * level_bar_length + "░" * (40 - level_bar_length)
                                print(f"📊 Input Level: |{level_bar}|")
                                
                                print("📈 All Class Probabilities:")
                                
                                # Display extended probabilities including Silence
                                extended_classes = ['Silence'] + TARGET_CLASSES
                                prob_pairs = [(extended_classes[i], extended_probs[i]) for i in range(len(extended_classes))]
                                prob_pairs.sort(key=lambda x: x[1], reverse=True)
                                
                                for class_name, prob in prob_pairs:
                                    bar_length = int(prob * 30)
                                    bar = "█" * bar_length + "░" * (30 - bar_length)
                                    icon = "🔇" if class_name == "Silence" else "🔊"
                                    print(f"  {icon} {class_name:<15}: {prob:.3f} |{bar}|")
                                
                                # Show VAD debug info for activity
                                if is_activity and vad_debug:
                                    print(f"🔍 VAD Details: Energy={vad_debug['energy']:.4f}, "
                                          f"Spectral={vad_debug['spectral_centroid']:.1f}Hz, "
                                          f"ZCR={vad_debug['zcr']:.4f}")
                                
                                print("-" * 80)
                            else:
                                # Show quick update with VAD and audio level
                                vad_icon = "🔊" if is_activity else "🔇"
                                level_indicator = "●" if audio_rms > 0.01 else "○"
                                print(f"\r[{timestamp}] {vad_icon} {predicted_class:<12} ({confidence:.3f}) {level_indicator}", end='', flush=True)
                        
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
            
            # VAD Statistics
            if 'vad_activity' in df.columns:
                vad_activity_count = df['vad_activity'].sum()
                vad_silence_count = len(df) - vad_activity_count
                print(f"\nVAD Statistics:")
                print(f"  Activity detected: {vad_activity_count} samples ({vad_activity_count/total_predictions*100:.1f}%)")
                print(f"  Silence detected: {vad_silence_count} samples ({vad_silence_count/total_predictions*100:.1f}%)")
            
            print("\nClass Distribution:")
            
            # Include Silence in the summary
            extended_classes = ['Silence'] + TARGET_CLASSES
            for class_name in extended_classes:
                count = class_counts.get(class_name, 0)
                percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
                bar_length = int(percentage / 100 * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                icon = "🔇" if class_name == "Silence" else "🔊"
                print(f"  {icon} {class_name:<15}: {count:3d} ({percentage:5.1f}%) |{bar}|")
            
            # Average confidence per class
            print("\nAverage Confidence by Class:")
            for class_name in extended_classes:
                class_data = df[df['predicted_class'] == class_name]
                if len(class_data) > 0:
                    avg_conf = class_data['confidence'].mean()
                    icon = "🔇" if class_name == "Silence" else "🔊"
                    print(f"  {icon} {class_name:<15}: {avg_conf:.3f}")
                else:
                    icon = "🔇" if class_name == "Silence" else "🔊"
                    print(f"  {icon} {class_name:<15}: No predictions")
            
            print("="*60)