"""
Flask Server for Real-Time Audio Classification with Firebase Integration
Receives audio data from clients, performs inference, and pushes results to Firebase
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import base64
import io
import wave
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime
import os
import sys

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from NoiseAnalyzer import NoiseAnalyzer, TARGET_CLASSES, SAMPLE_RATE, DURATION
from VAD import VoiceActivityDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Firebase
firebase_config = {
    "apiKey": "AIzaSyDyH0LXgzkikiCTxgsw0ebEmjjQ0vkOl-w",
    "authDomain": "hawties-2a013.firebaseapp.com",
    "projectId": "hawties-2a013",
    "storageBucket": "hawties-2a013.firebasestorage.app",
    "messagingSenderId": "523392422092",
    "appId": "1:523392422092:web:f63642ccc26c3888a1b269",
    "measurementId": "G-JCWEDXQL6W",
}
# Initialize Firebase Admin SDK
# Note: You'll need to create a service account key from Firebase Console
# and save it as 'firebase-credentials.json' in the project directory
try:
    if os.path.exists("firebase-credentials.json"):
        cred = credentials.Certificate("firebase-credentials.json")
    else:
        # Fallback: Try to use environment variables or default credentials
        print(
            "Warning: firebase-credentials.json not found. Using default credentials."
        )
        cred = None

    if cred:
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()

    db = firestore.client()
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    print(
        "Note: You need to download service account credentials from Firebase Console"
    )
    print("Go to: Project Settings > Service Accounts > Generate new private key")
    print("Save it as 'firebase-credentials.json' in the project directory")
    db = None

# Initialize the analyzer and VAD
analyzer = NoiseAnalyzer()
vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

# Load the trained model
MODEL_PATH = "app/models/best_model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please train a model first or update the MODEL_PATH")
else:
    try:
        analyzer.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Global state for VAD calibration
vad_calibration_complete = False
calibration_samples_needed = int(3.5 / DURATION)  # Number of 1-second samples needed


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": analyzer.model is not None,
            "vad_calibrated": vad.is_calibrated,
            "firebase_connected": db is not None,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/calibrate", methods=["POST"])
def calibrate_vad():
    """Calibrate the VAD with silence samples"""
    global vad_calibration_complete

    try:
        # Get audio data from request
        data = request.json
        if not data or "audio" not in data:
            return jsonify({"error": "No audio data provided"}), 400

        # Decode audio data (base64 encoded)
        audio_bytes = base64.b64decode(data["audio"])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Resample if needed
        if "sample_rate" in data and data["sample_rate"] != SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array, orig_sr=data["sample_rate"], target_sr=SAMPLE_RATE
            )

        # Ensure correct length
        target_length = int(DURATION * SAMPLE_RATE)
        if len(audio_array) > target_length:
            audio_array = audio_array[:target_length]
        elif len(audio_array) < target_length:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))

        # Add calibration sample
        calibration_complete = vad.add_calibration_sample(audio_array)
        progress = vad.get_calibration_progress()

        if calibration_complete:
            vad_calibration_complete = True
            return jsonify(
                {
                    "status": "complete",
                    "progress": 100.0,
                    "message": "VAD calibration complete",
                }
            )
        else:
            return jsonify(
                {
                    "status": "in_progress",
                    "progress": progress,
                    "message": f"Calibration {progress:.1f}% complete",
                }
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict audio class and push to Firebase

    Expects JSON with:
    - audio: base64 encoded audio data (float32 array)
    - sample_rate: original sample rate (optional, defaults to 16000)
    - device_id: unique identifier for the device (optional)
    """
    global vad_calibration_complete

    try:
        # Get audio data from request
        data = request.json
        if not data or "audio" not in data:
            return jsonify({"error": "No audio data provided"}), 400

        # Check if VAD is calibrated
        if not vad.is_calibrated:
            return (
                jsonify(
                    {
                        "error": "VAD not calibrated",
                        "message": "Please calibrate the VAD first by sending silence samples to /calibrate",
                        "calibration_progress": vad.get_calibration_progress(),
                    }
                ),
                400,
            )

        # Decode audio data (base64 encoded float32 array)
        audio_bytes = base64.b64decode(data["audio"])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Resample if needed
        original_sr = data.get("sample_rate", SAMPLE_RATE)
        if original_sr != SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array, orig_sr=original_sr, target_sr=SAMPLE_RATE
            )

        # Ensure correct length (1 second)
        target_length = int(DURATION * SAMPLE_RATE)
        if len(audio_array) > target_length:
            audio_array = audio_array[:target_length]
        elif len(audio_array) < target_length:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))

        # Perform VAD
        is_activity, vad_confidence, vad_debug = vad.detect_activity(audio_array)

        # Perform classification
        if is_activity:
            predicted_class, confidence, all_probs = analyzer.predict_audio(audio_array)
        else:
            predicted_class = "Silence"
            confidence = vad_confidence
            all_probs = np.zeros(len(TARGET_CLASSES))

        # Calculate audio statistics
        audio_rms = float(np.sqrt(np.mean(audio_array**2)))
        audio_max = float(np.max(np.abs(audio_array)))

        # Prepare result
        timestamp = datetime.now()
        device_id = data.get("device_id", "unknown")

        # Create extended classes including Silence
        extended_classes = ["Silence"] + TARGET_CLASSES
        extended_probs = np.zeros(len(extended_classes))

        if predicted_class == "Silence":
            extended_probs[0] = confidence
        else:
            extended_probs[0] = 1 - vad_confidence
            for i, prob in enumerate(all_probs):
                if i + 1 < len(extended_probs):
                    extended_probs[i + 1] = float(prob)

        result = {
            "timestamp": timestamp.isoformat(),
            "device_id": device_id,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "vad_activity": bool(is_activity),
            "vad_confidence": float(vad_confidence),
            "probabilities": {
                extended_classes[i]: float(extended_probs[i])
                for i in range(len(extended_classes))
            },
            "audio_stats": {"rms": audio_rms, "max": audio_max},
        }

        # Add VAD debug info if available
        if isinstance(vad_debug, dict):
            result["vad_debug"] = {
                k: (
                    float(v)
                    if isinstance(v, (int, float, np.number))
                    else bool(v) if isinstance(v, (bool, np.bool_)) else v
                )
                for k, v in vad_debug.items()
            }

        # Push to Firebase
        if db is not None:
            try:
                # Add to 'predictions' collection
                doc_ref = db.collection("predictions").add(
                    {
                        "timestamp": timestamp,
                        "device_id": device_id,
                        "predicted_class": predicted_class,
                        "confidence": float(confidence),
                        "vad_activity": bool(is_activity),
                        "vad_confidence": float(vad_confidence),
                        "probabilities": result["probabilities"],
                        "audio_stats": result["audio_stats"],
                    }
                )

                result["firebase_id"] = doc_ref[1].id
                print(f"✅ Pushed to Firebase: {predicted_class} ({confidence:.3f})")
            except Exception as firebase_error:
                print(f"❌ Firebase error: {firebase_error}")
                result["firebase_error"] = str(firebase_error)
        else:
            result["firebase_error"] = "Firebase not initialized"

        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/reset_vad", methods=["POST"])
def reset_vad():
    """Reset VAD calibration"""
    global vad_calibration_complete

    vad.reset_calibration()
    vad_calibration_complete = False

    return jsonify(
        {
            "status": "success",
            "message": "VAD calibration reset. Please recalibrate with silence samples.",
        }
    )


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get statistics from Firebase"""
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500

    try:
        # Get recent predictions (last 100)
        predictions_ref = (
            db.collection("predictions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(100)
        )

        predictions = predictions_ref.stream()

        # Count by class
        class_counts = {cls: 0 for cls in ["Silence"] + TARGET_CLASSES}
        total_count = 0

        for pred in predictions:
            pred_data = pred.to_dict()
            predicted_class = pred_data.get("predicted_class", "Unknown")
            if predicted_class in class_counts:
                class_counts[predicted_class] += 1
            total_count += 1

        return jsonify(
            {
                "total_predictions": total_count,
                "class_distribution": class_counts,
                "percentages": {
                    cls: (count / total_count * 100) if total_count > 0 else 0
                    for cls, count in class_counts.items()
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 70)
    print("🎙️  NOISE ANALYZER FLASK SERVER")
    print("=" * 70)
    print(f"Model loaded: {'✅' if analyzer.model is not None else '❌'}")
    print(f"Firebase connected: {'✅' if db is not None else '❌'}")
    print(f"VAD calibrated: {'✅' if vad.is_calibrated else '❌ (needs calibration)'}")
    print()
    print("Endpoints:")
    print("  GET  /health       - Health check")
    print("  POST /calibrate    - Calibrate VAD with silence samples")
    print("  POST /predict      - Predict audio class")
    print("  POST /reset_vad    - Reset VAD calibration")
    print("  GET  /stats        - Get prediction statistics from Firebase")
    print()
    print("Starting server on http://0.0.0.0:5000")
    print("=" * 70)

    # Run the Flask app
    app.run(host="0.0.0.0", port=6002, debug=False)
