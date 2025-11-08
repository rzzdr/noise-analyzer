#!/usr/bin/env python3
"""
ESP32 Noise Monitor Logger
=========================

Real-time serial data capture and visualization for ESP32-CAM noise classifier.
Features:
- Auto-detect ESP32 COM port
- Real-time console display with emoji indicators
- CSV logging with timestamps
- Live probability bars and confidence visualization
- Inference timing statistics
- Memory usage monitoring
- Confusion matrix tracking (when ground truth provided)

Usage:
    python esp32_logger.py [--port COM3] [--baudrate 115200] [--output data.csv]

Author: AI Assistant
"""

import serial
import serial.tools.list_ports
import argparse
import csv
import time
import threading
import queue
import os
import sys
from datetime import datetime
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import json

# Class definitions
CLASS_NAMES = ['Whispering', 'Typing', 'Phone_ringing', 'Loud_talking']
CLASS_EMOJIS = ['💬', '⌨️', '📞', '🗣️']

@dataclass
class ClassificationData:
    timestamp: int
    class_name: str
    confidence: float
    inference_time_ms: int
    vad_confidence: float
    heap_free: int
    psram_free: int
    actual_timestamp: datetime
    
    @classmethod
    def from_csv_line(cls, line: str) -> Optional['ClassificationData']:
        """Parse CSV line from ESP32 output"""
        try:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                return cls(
                    timestamp=int(parts[0]),
                    class_name=parts[1].strip(),
                    confidence=float(parts[2]),
                    inference_time_ms=int(parts[3]),
                    vad_confidence=float(parts[4]),
                    heap_free=int(parts[5]),
                    psram_free=int(parts[6]),
                    actual_timestamp=datetime.now()
                )
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line.strip()} - {e}")
        return None

class InferenceStats:
    """Track inference timing statistics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.vad_confidences = deque(maxlen=window_size)
        self.class_counts = defaultdict(int)
        
    def add_sample(self, data: ClassificationData):
        if data.class_name != "Silence":
            self.inference_times.append(data.inference_time_ms)
        self.vad_confidences.append(data.vad_confidence)
        self.class_counts[data.class_name] += 1
        
    def get_stats(self) -> Dict:
        if not self.inference_times:
            return {
                'avg_time': 0,
                'min_time': 0,
                'max_time': 0,
                'std_time': 0,
                'avg_vad_confidence': 0,
                'total_samples': sum(self.class_counts.values())
            }
            
        times = list(self.inference_times)
        vad_confs = list(self.vad_confidences)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'avg_vad_confidence': np.mean(vad_confs),
            'total_samples': sum(self.class_counts.values())
        }

class ESP32Logger:
    """Main logger class for ESP32 serial communication"""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, 
                 output_file: str = "esp32_log.csv"):
        self.port = port
        self.baudrate = baudrate
        self.output_file = output_file
        self.serial_conn = None
        self.is_running = False
        self.data_queue = queue.Queue()
        self.stats = InferenceStats()
        
        # Recent data for display
        self.recent_data = deque(maxlen=10)
        self.class_probabilities = {name: 0.0 for name in CLASS_NAMES + ["Silence"]}
        
        # CSV writer
        self.csv_file = None
        self.csv_writer = None
        
        # Manual labeling for confusion matrix
        self.manual_labels = deque(maxlen=1000)
        self.waiting_for_label = False
        self.current_prediction = None
        
    def find_esp32_port(self) -> Optional[str]:
        """Auto-detect ESP32 COM port"""
        print("🔍 Scanning for ESP32 devices...")
        
        ports = serial.tools.list_ports.comports()
        esp32_ports = []
        
        for port in ports:
            # Look for common ESP32 identifiers
            if any(keyword in port.description.lower() for keyword in 
                   ['cp210', 'ch340', 'ftdi', 'silicon labs', 'esp32']):
                esp32_ports.append(port.device)
                print(f"   Found potential ESP32: {port.device} - {port.description}")
        
        if not esp32_ports:
            print("❌ No ESP32 devices found")
            print("Available ports:")
            for port in ports:
                print(f"   {port.device} - {port.description}")
            return None
        
        if len(esp32_ports) == 1:
            selected_port = esp32_ports[0]
            print(f"✅ Auto-selected: {selected_port}")
            return selected_port
        
        # Multiple ports found - let user choose
        print(f"🔢 Multiple ESP32 devices found:")
        for i, port in enumerate(esp32_ports):
            print(f"   {i+1}: {port}")
        
        try:
            choice = int(input("Select port number: ")) - 1
            if 0 <= choice < len(esp32_ports):
                return esp32_ports[choice]
        except ValueError:
            pass
        
        print("❌ Invalid selection")
        return None
    
    def connect(self) -> bool:
        """Connect to ESP32 serial port"""
        if self.port is None:
            self.port = self.find_esp32_port()
            if self.port is None:
                return False
        
        try:
            print(f"🔌 Connecting to {self.port} at {self.baudrate} baud...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            # Flush any existing data
            self.serial_conn.reset_input_buffer()
            
            print(f"✅ Connected to ESP32")
            return True
            
        except serial.SerialException as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("📴 Disconnected from ESP32")
    
    def open_csv_file(self):
        """Open CSV file for logging"""
        try:
            self.csv_file = open(self.output_file, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            header = ['actual_timestamp', 'esp32_timestamp', 'class_name', 'confidence', 
                     'inference_time_ms', 'vad_confidence', 'heap_free_kb', 'psram_free_kb']
            self.csv_writer.writerow(header)
            
            print(f"📝 Logging to: {self.output_file}")
            
        except IOError as e:
            print(f"❌ Failed to open CSV file: {e}")
            return False
        
        return True
    
    def close_csv_file(self):
        """Close CSV file"""
        if self.csv_file:
            self.csv_file.close()
            print(f"💾 CSV file saved: {self.output_file}")
    
    def read_serial_data(self):
        """Read data from serial port in background thread"""
        print("📡 Starting serial data reader...")
        
        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                
                if line:
                    # Try to parse as classification data
                    data = ClassificationData.from_csv_line(line)
                    if data:
                        self.data_queue.put(('data', data))  
                    else:
                        # Pass through other messages (debug, status, etc.)
                        self.data_queue.put(('message', line))
                        
            except serial.SerialException as e:
                print(f"❌ Serial read error: {e}")
                break
            except UnicodeDecodeError:
                # Skip invalid characters
                continue
        
        print("📡 Serial reader stopped")
    
    def process_data(self, data: ClassificationData):
        """Process received classification data"""
        # Update statistics
        self.stats.add_sample(data)
        self.recent_data.append(data)
        
        # Update probability display
        if data.class_name == "Silence":
            # Reset all probabilities and set silence
            for name in CLASS_NAMES:
                self.class_probabilities[name] = 0.0
            self.class_probabilities["Silence"] = data.vad_confidence
        else:
            # Set current class probability, reset others
            for name in CLASS_NAMES:
                self.class_probabilities[name] = 0.0
            self.class_probabilities["Silence"] = 0.0
            
            if data.class_name in self.class_probabilities:
                self.class_probabilities[data.class_name] = data.confidence
        
        # Log to CSV
        if self.csv_writer:
            row = [
                data.actual_timestamp.isoformat(),
                data.timestamp,
                data.class_name,
                data.confidence,
                data.inference_time_ms,
                data.vad_confidence,
                data.heap_free // 1024,  # Convert to KB
                data.psram_free // 1024
            ]
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # Ensure data is written immediately
        
        # Display result
        self.display_classification(data)
        
        # Handle manual labeling
        if data.class_name != "Silence" and not self.waiting_for_label:
            self.current_prediction = data
            self.waiting_for_label = True
    
    def display_classification(self, data: ClassificationData):
        """Display classification result in console"""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        if data.class_name == "Silence":
            print(f"[{current_time}] 🔇 SILENCE (VAD: {data.vad_confidence:.2f})")
        else:
            # Find emoji for class
            emoji = "🎯"
            if data.class_name in CLASS_NAMES:
                emoji = CLASS_EMOJIS[CLASS_NAMES.index(data.class_name)]
            
            print(f"[{current_time}] {emoji} {data.class_name.upper()} "
                  f"(Conf: {data.confidence:.2f}, Time: {data.inference_time_ms}ms)")
        
        # Show probability bars every few classifications
        if len(self.recent_data) % 3 == 0:  # Every 3rd classification
            self.display_probability_bars()
    
    def display_probability_bars(self):
        """Display current class probabilities as ASCII bars"""
        print("📊 Current Probabilities:")
        
        # Sort by probability for better display
        sorted_probs = sorted(self.class_probabilities.items(), 
                            key=lambda x: x[1], reverse=True)
        
        for class_name, prob in sorted_probs:
            if prob > 0.01:  # Only show non-zero probabilities
                # Find emoji
                emoji = "🔇" if class_name == "Silence" else "🎯"
                if class_name in CLASS_NAMES:
                    emoji = CLASS_EMOJIS[CLASS_NAMES.index(class_name)]
                
                # Create ASCII bar
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                
                print(f"   {emoji} {class_name:12} {prob:5.2f} |{bar}|")
        
        # Show recent statistics
        stats = self.stats.get_stats()
        if stats['total_samples'] > 0:
            print(f"⚡ Stats (last {len(self.stats.inference_times)}): "
                  f"Avg={stats['avg_time']:.0f}ms, "
                  f"Min={stats['min_time']:.0f}ms, "
                  f"Max={stats['max_time']:.0f}ms")
            
            # Memory info
            if self.recent_data:
                latest = self.recent_data[-1]
                print(f"💾 Memory: Heap={latest.heap_free//1024}KB, "
                      f"PSRAM={latest.psram_free//1024}KB")
        
        print()  # Empty line for readability
    
    def handle_manual_labeling(self):
        """Handle manual labeling in separate thread"""
        print("\n" + "="*60)
        print("MANUAL LABELING MODE")
        print("="*60)
        print("When a classification appears, you can provide the ground truth:")
        print("0: Whispering  1: Typing  2: Phone_ringing  3: Loud_talking")
        print("s: Skip  q: Quit labeling")
        print("="*60)
        
        while self.is_running:
            if self.waiting_for_label and self.current_prediction:
                try:
                    # Prompt for ground truth
                    pred_data = self.current_prediction
                    print(f"\n🏷️  ESP32 predicted: {pred_data.class_name} ({pred_data.confidence:.2f})")
                    label_input = input("Ground truth (0-3/s/q): ").strip().lower()
                    
                    if label_input == 'q':
                        print("Exiting manual labeling mode")
                        break
                    elif label_input == 's':
                        print("Skipped")
                    elif label_input.isdigit() and 0 <= int(label_input) <= 3:
                        ground_truth = CLASS_NAMES[int(label_input)]
                        
                        # Store for confusion matrix
                        self.manual_labels.append({
                            'predicted': pred_data.class_name,
                            'actual': ground_truth,
                            'confidence': pred_data.confidence,
                            'timestamp': pred_data.actual_timestamp
                        })
                        
                        # Show immediate feedback
                        correct = (pred_data.class_name == ground_truth)
                        status = "✅ CORRECT" if correct else "❌ INCORRECT"
                        print(f"{status} - Actual: {ground_truth}")
                    else:
                        print("Invalid input")
                    
                    self.waiting_for_label = False
                    self.current_prediction = None
                    
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
            
            time.sleep(0.1)
    
    def generate_confusion_matrix(self):
        """Generate confusion matrix from manual labels"""
        if len(self.manual_labels) < 5:
            print(f"Not enough manual labels ({len(self.manual_labels)}) for confusion matrix")
            return
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame(self.manual_labels)
        
        # Create confusion matrix
        actual_labels = df['actual'].tolist()
        predicted_labels = df['predicted'].tolist()
        
        # Create matrix
        matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
        
        for actual, predicted in zip(actual_labels, predicted_labels):
            if actual in CLASS_NAMES and predicted in CLASS_NAMES:
                actual_idx = CLASS_NAMES.index(actual)
                predicted_idx = CLASS_NAMES.index(predicted)
                matrix[actual_idx][predicted_idx] += 1
        
        # Print confusion matrix
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print("Rows: Actual, Columns: Predicted")
        print()
        
        # Header
        print("Actual \\ Predicted", end="")
        for name in CLASS_NAMES:
            print(f"{name[:8]:>10}", end="")
        print(f"{'Accuracy':>10}")
        
        # Matrix rows
        total_correct = 0
        total_samples = 0
        
        for i, actual_name in enumerate(CLASS_NAMES):
            print(f"{actual_name[:15]:15}", end="")
            
            row_sum = sum(matrix[i])
            correct = matrix[i][i]
            
            for j in range(len(CLASS_NAMES)):
                print(f"{matrix[i][j]:10}", end="")
            
            accuracy = (correct / row_sum * 100) if row_sum > 0 else 0
            print(f"{accuracy:9.1f}%")
            
            total_correct += correct
            total_samples += row_sum
        
        # Overall accuracy
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        print(f"\nOverall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_samples})")
        
        # Save confusion matrix data
        confusion_file = self.output_file.replace('.csv', '_confusion.json')
        confusion_data = {
            'matrix': matrix.tolist(),
            'class_names': CLASS_NAMES,
            'total_samples': int(total_samples),
            'overall_accuracy': float(overall_accuracy),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(confusion_file, 'w') as f:
            json.dump(confusion_data, f, indent=2)
        
        print(f"Confusion matrix saved: {confusion_file}")
    
    def run(self, enable_manual_labeling: bool = False):
        """Main run loop"""
        if not self.connect():
            return False
        
        if not self.open_csv_file():
            return False
        
        self.is_running = True
        
        # Start serial reader thread
        serial_thread = threading.Thread(target=self.read_serial_data, daemon=True)
        serial_thread.start()
        
        # Start manual labeling thread if requested
        labeling_thread = None
        if enable_manual_labeling:
            labeling_thread = threading.Thread(target=self.handle_manual_labeling, daemon=True)
            labeling_thread.start()
        
        print("\n" + "="*60)
        print("🚀 ESP32 NOISE MONITOR LOGGER STARTED")
        print("="*60)
        print(f"Port: {self.port}")
        print(f"Baudrate: {self.baudrate}")
        print(f"Output: {self.output_file}")
        print(f"Manual labeling: {'Enabled' if enable_manual_labeling else 'Disabled'}")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        try:
            # Main processing loop
            while self.is_running:
                try:
                    # Process queued data
                    msg_type, content = self.data_queue.get(timeout=1.0)
                    
                    if msg_type == 'data':
                        self.process_data(content)
                    elif msg_type == 'message':
                        # Print non-data messages (debug, status, etc.)
                        if content and not content.startswith('timestamp,'):  # Skip CSV header
                            print(f"ESP32: {content}")
                            
                except queue.Empty:
                    continue
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping logger...")
        
        finally:
            self.is_running = False
            
            # Wait a bit for threads to finish
            time.sleep(1)
            
            # Generate final reports
            if enable_manual_labeling and len(self.manual_labels) > 0:
                self.generate_confusion_matrix()
            
            # Print final statistics
            self.print_final_stats()
            
            # Cleanup
            self.close_csv_file()
            self.disconnect()
        
        return True
    
    def print_final_stats(self):
        """Print final statistics"""
        stats = self.stats.get_stats()
        
        print("\n" + "="*60)
        print("📈 FINAL STATISTICS")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        
        if stats['total_samples'] > 0:
            print(f"Inference timing:")
            print(f"  Average: {stats['avg_time']:.1f}ms")
            print(f"  Min: {stats['min_time']:.1f}ms")
            print(f"  Max: {stats['max_time']:.1f}ms")
            print(f"  Std Dev: {stats['std_time']:.1f}ms")
            
            print(f"Average VAD confidence: {stats['avg_vad_confidence']:.3f}")
            
            print("\nClass distribution:")
            total_classifications = sum(self.stats.class_counts.values())
            for class_name, count in sorted(self.stats.class_counts.items(), 
                                          key=lambda x: x[1], reverse=True):
                percentage = (count / total_classifications * 100) if total_classifications > 0 else 0
                emoji = "🔇" if class_name == "Silence" else "🎯"
                if class_name in CLASS_NAMES:
                    emoji = CLASS_EMOJIS[CLASS_NAMES.index(class_name)]
                print(f"  {emoji} {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nData saved to: {self.output_file}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='ESP32 Noise Monitor Data Logger')
    parser.add_argument('--port', help='Serial port (auto-detect if not specified)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--output', default='esp32_log.csv', help='Output CSV file (default: esp32_log.csv)')
    parser.add_argument('--manual-labeling', action='store_true', 
                       help='Enable manual labeling for confusion matrix generation')
    
    args = parser.parse_args()
    
    # Create timestamped output filename if using default
    if args.output == 'esp32_log.csv':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'esp32_log_{timestamp}.csv'
    
    logger = ESP32Logger(
        port=args.port,
        baudrate=args.baudrate,
        output_file=args.output
    )
    
    success = logger.run(enable_manual_labeling=args.manual_labeling)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())