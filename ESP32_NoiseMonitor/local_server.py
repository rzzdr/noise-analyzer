#!/usr/bin/env python3
"""
Local Flask Audio Server for ESP32 Testing
==========================================

A simple local Flask server that mimics the remote server
for testing ESP32 connectivity and audio streaming.

Run this on your local machine and update ESP32 config to point here.
"""

from flask import Flask, request, jsonify
import base64
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

# Calibration state
calibration_samples = 0
calibration_complete = False

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "ESP32 Local Audio Server Running",
        "version": "1.0.0",
        "endpoints": ["/predict", "/calibrate"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio prediction requests from ESP32"""
    try:
        data = request.get_json()
        
        if not data or 'audio' not in data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode base64 audio
        audio_base64 = data['audio']
        device_id = data.get('device_id', 'unknown')
        sample_rate = data.get('sample_rate', 16000)
        
        # Decode audio data
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Simple analysis
        rms = float(np.sqrt(np.mean(audio_array**2)))
        max_val = float(np.max(np.abs(audio_array)))
        
        # Mock prediction based on RMS level
        if rms > 0.1:
            predicted_class = "Loud_talking"
            confidence = min(0.9, rms * 2)
        elif rms > 0.05:
            predicted_class = "Whispering" 
            confidence = min(0.8, rms * 4)
        elif rms > 0.01:
            predicted_class = "Typing"
            confidence = min(0.7, rms * 10)
        else:
            predicted_class = "Silence"
            confidence = 0.6
        
        # Create response
        response = {
            "device_id": device_id,
            "timestamp": datetime.now().isoformat(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "audio_stats": {
                "rms": rms,
                "max": max_val,
                "samples": len(audio_array),
                "sample_rate": sample_rate
            },
            "vad_activity": rms > 0.02,
            "vad_confidence": confidence,
            "probabilities": {
                "Silence": 0.2 if predicted_class != "Silence" else confidence,
                "Whispering": 0.2 if predicted_class != "Whispering" else confidence,
                "Typing": 0.2 if predicted_class != "Typing" else confidence,
                "Loud_talking": 0.2 if predicted_class != "Loud_talking" else confidence,
                "Phone_ringing": 0.2
            }
        }
        
        print(f"🎯 Prediction: {device_id} -> {predicted_class} ({confidence:.2f}) [RMS: {rms:.4f}]")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """Handle calibration requests from ESP32"""
    global calibration_samples, calibration_complete
    
    try:
        data = request.get_json()
        
        if not data or 'audio' not in data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode base64 audio
        audio_base64 = data['audio']
        device_id = data.get('device_id', 'unknown')
        
        # Decode audio data
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        calibration_samples += 1
        
        # Simple analysis for calibration
        rms = float(np.sqrt(np.mean(audio_array**2)))
        
        # Mock calibration progress
        progress = min(100.0, calibration_samples * 50.0)  # 2 samples = 100%
        
        if calibration_samples >= 2:
            calibration_complete = True
            status = "complete"
        else:
            status = "collecting"
        
        response = {
            "status": status,
            "progress": progress,
            "message": f"VAD calibration sample {calibration_samples} processed",
            "device_id": device_id,
            "audio_stats": {
                "rms": rms,
                "samples": len(audio_array)
            }
        }
        
        print(f"🎙️  Calibration: {device_id} -> Sample {calibration_samples} [RMS: {rms:.4f}] ({progress:.1f}%)")
        
        if calibration_complete:
            response["message"] = "VAD calibration complete"
            print(f"✅ Calibration completed for {device_id}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Calibration error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("╔═══════════════════════════════════════╗")
    print("║      ESP32 Local Audio Server        ║")
    print("║         Flask Testing Server         ║")
    print("╚═══════════════════════════════════════╝")
    print()
    print("🌐 Server will be available at:")
    print("   http://localhost:6002")
    print("   http://127.0.0.1:6002")
    print("   http://[your-local-ip]:6002")
    print()
    print("📝 To use with ESP32:")
    print('   1. Find your local IP address')
    print('   2. Update config.h:')
    print('      #define SERVER_URL "http://[your-ip]:6002/predict"')
    print('      #define CALIBRATE_URL "http://[your-ip]:6002/calibrate"')
    print()
    print("🚀 Starting server...")
    
    app.run(host='0.0.0.0', port=6002, debug=True)