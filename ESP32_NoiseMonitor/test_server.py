#!/usr/bin/env python3
"""
Test ESP32 Flask Server Connectivity
====================================

This script tests the Flask server endpoints to help debug
ESP32 calibration issues.
"""

import requests
import json
import base64
import numpy as np

# Server configuration
SERVER_URL = "http://4.240.35.54:6002"
PREDICT_URL = f"{SERVER_URL}/predict"
CALIBRATE_URL = f"{SERVER_URL}/calibrate"

def test_server_connectivity():
    """Test basic server connectivity"""
    print("🌐 Testing server connectivity...")
    
    try:
        response = requests.get(SERVER_URL, timeout=5)
        print(f"   Server response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    return True

def test_predict_endpoint():
    """Test the /predict endpoint with dummy audio data"""
    print("\n🎯 Testing /predict endpoint...")
    
    # Create dummy audio data (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0  # seconds
    samples = int(sample_rate * duration)
    
    # Generate dummy audio (sine wave)
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples, False)).astype(np.float32)
    
    # Convert to bytes and base64 encode
    audio_bytes = audio_data.tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Create JSON payload
    payload = {
        "audio": audio_base64,
        "sample_rate": sample_rate,
        "device_id": "TEST_DEVICE"
    }
    
    try:
        response = requests.post(PREDICT_URL, json=payload, timeout=15)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")

def test_calibrate_endpoint():
    """Test the /calibrate endpoint with dummy audio data"""
    print("\n🎙️  Testing /calibrate endpoint...")
    
    # Create dummy audio data (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0  # seconds
    samples = int(sample_rate * duration)
    
    # Generate dummy audio (low amplitude noise for calibration)
    audio_data = (np.random.normal(0, 0.01, samples)).astype(np.float32)
    
    # Convert to bytes and base64 encode
    audio_bytes = audio_data.tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Create JSON payload
    payload = {
        "audio": audio_base64,
        "sample_rate": sample_rate,
        "device_id": "TEST_DEVICE"
    }
    
    try:
        response = requests.post(CALIBRATE_URL, json=payload, timeout=15)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")

def main():
    print("╔═══════════════════════════════════════╗")
    print("║      ESP32 Server Connectivity Test   ║")
    print("╚═══════════════════════════════════════╝")
    
    # Test basic connectivity
    if not test_server_connectivity():
        print("\n❌ Server not accessible. Check server status.")
        return
    
    # Test endpoints
    test_predict_endpoint()
    test_calibrate_endpoint()
    
    print("\n✅ Server testing completed!")

if __name__ == "__main__":
    main()