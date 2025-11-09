"""
Test Client for Flask Audio Classification Server
Captures audio from laptop microphone and sends to server for inference
"""

import numpy as np
import sounddevice as sd
import requests
import base64
import time
import argparse
from datetime import datetime
import sys

# Configuration
SERVER_URL = "http://4.240.35.54:6002"  # Change to your server IP if not local
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second audio chunks
DEVICE_ID = "test_laptop"  # Unique identifier for this client


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_colored(text, color):
    """Print colored text"""
    print(f"{color}{text}{Colors.ENDC}")


def check_server_health(server_url):
    """Check if server is healthy"""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print_colored("✅ Server health check passed", Colors.OKGREEN)
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   VAD calibrated: {health_data.get('vad_calibrated', False)}")
            print(
                f"   Firebase connected: {health_data.get('firebase_connected', False)}"
            )
            return True, health_data
        else:
            print_colored(
                f"❌ Server returned status {response.status_code}", Colors.FAIL
            )
            return False, None
    except requests.exceptions.ConnectionError:
        print_colored(f"❌ Cannot connect to server at {server_url}", Colors.FAIL)
        print(f"   Make sure the server is running!")
        return False, None
    except Exception as e:
        print_colored(f"❌ Health check failed: {e}", Colors.FAIL)
        return False, None


def list_audio_devices():
    """List available audio input devices"""
    print("\n📱 Available audio input devices:")
    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"   {i}: {device['name']}{default_marker}")
            print(
                f"      Max channels: {device['max_input_channels']}, "
                f"Default SR: {device['default_samplerate']}"
            )
            input_devices.append(i)

    return input_devices


def test_microphone(device_id=None, duration=3):
    """Test microphone for specified duration"""
    print_colored(f"\n🎤 Testing microphone for {duration} seconds...", Colors.OKCYAN)
    print("   Please make some noise to verify audio input is working...")

    test_audio = []

    def callback(indata, frames, time, status):
        if status:
            print(f"   Status: {status}")
        test_audio.extend(indata[:, 0])

    try:
        with sd.InputStream(
            callback=callback, device=device_id, channels=1, samplerate=SAMPLE_RATE
        ):
            for i in range(duration):
                time.sleep(1)
                if len(test_audio) > SAMPLE_RATE:
                    recent_audio = np.array(test_audio[-SAMPLE_RATE:])
                    rms = np.sqrt(np.mean(recent_audio**2))
                    level_bar_length = min(int(rms * 1000), 40)
                    level_bar = "█" * level_bar_length + "░" * (40 - level_bar_length)
                    print(f"   [{i+1}/{duration}] |{level_bar}| RMS={rms:.4f}")

        if len(test_audio) == 0:
            print_colored("   ⚠️  WARNING: No audio detected!", Colors.WARNING)
            return False
        else:
            final_rms = np.sqrt(np.mean(np.array(test_audio) ** 2))
            if final_rms < 0.001:
                print_colored(
                    "   ⚠️  WARNING: Very low audio levels detected!", Colors.WARNING
                )
                return False
            else:
                print_colored("   ✅ Microphone test passed!", Colors.OKGREEN)
                return True

    except Exception as e:
        print_colored(f"   ❌ Microphone test failed: {e}", Colors.FAIL)
        return False


def calibrate_vad(server_url, device_id=None, duration=3.5):
    """Calibrate VAD on server with silence samples"""
    print_colored(
        f"\n🔇 VAD Calibration Phase (keep quiet for {duration} seconds)", Colors.HEADER
    )

    samples_needed = int(duration / DURATION)

    for i in range(samples_needed):
        # Record 1 second of silence
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device_id,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        # Send to server
        audio_bytes = audio.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        try:
            response = requests.post(
                f"{server_url}/calibrate",
                json={"audio": audio_b64, "sample_rate": SAMPLE_RATE},
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                progress = result.get("progress", 0)
                bar_length = int(progress / 100 * 40)
                progress_bar = "█" * bar_length + "░" * (40 - bar_length)
                print(
                    f"\r   Progress: |{progress_bar}| {progress:.1f}%",
                    end="",
                    flush=True,
                )

                if result.get("status") == "complete":
                    print()
                    print_colored("   ✅ VAD calibration complete!", Colors.OKGREEN)
                    return True
            else:
                print()
                print_colored(f"   ❌ Calibration failed: {response.text}", Colors.FAIL)
                return False

        except Exception as e:
            print()
            print_colored(f"   ❌ Calibration error: {e}", Colors.FAIL)
            return False

    print()
    return True


def send_audio_for_prediction(server_url, audio, device_id_str):
    """Send audio to server for prediction"""
    # Encode audio as base64
    audio_bytes = audio.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    try:
        response = requests.post(
            f"{server_url}/predict",
            json={
                "audio": audio_b64,
                "sample_rate": SAMPLE_RATE,
                "device_id": device_id_str,
            },
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()
        else:
            error_data = (
                response.json()
                if response.headers.get("content-type") == "application/json"
                else {"error": response.text}
            )
            return {"error": error_data.get("error", "Unknown error")}

    except Exception as e:
        return {"error": str(e)}


def start_real_time_testing(server_url, device_id_audio=None, device_id_str=DEVICE_ID):
    """Start real-time audio testing"""
    print_colored(f"\n🔊 Starting real-time audio classification", Colors.HEADER)
    print("   Press Ctrl+C to stop\n")
    print("=" * 80)

    prediction_count = 0

    try:
        while True:
            # Record 1 second of audio
            audio = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                device=device_id_audio,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()

            # Calculate audio level
            audio_rms = np.sqrt(np.mean(audio**2))

            # Send for prediction
            result = send_audio_for_prediction(server_url, audio, device_id_str)

            prediction_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if "error" in result:
                print_colored(f"[{timestamp}] ❌ Error: {result['error']}", Colors.FAIL)
            else:
                # Extract results
                predicted_class = result.get("predicted_class", "Unknown")
                confidence = result.get("confidence", 0.0)
                vad_activity = result.get("vad_activity", False)
                probabilities = result.get("probabilities", {})

                # Print results
                print(f"\n[{timestamp}] Prediction #{prediction_count}")

                # VAD status
                vad_icon = "🔊" if vad_activity else "🔇"
                vad_status = "Activity" if vad_activity else "Silence"
                print(f"{vad_icon} VAD: {vad_status}")

                # Classification
                class_color = Colors.OKGREEN if confidence > 0.7 else Colors.WARNING
                print_colored(
                    f"🎯 CLASSIFICATION: {predicted_class} (Confidence: {confidence:.3f})",
                    class_color,
                )

                # Audio level
                level_bar_length = min(int(audio_rms * 1000), 40)
                level_bar = "█" * level_bar_length + "░" * (40 - level_bar_length)
                print(f"🎤 Audio Level: |{level_bar}| RMS={audio_rms:.4f}")

                # Probabilities
                if probabilities:
                    print("📈 All Class Probabilities:")
                    prob_items = sorted(
                        probabilities.items(), key=lambda x: x[1], reverse=True
                    )
                    for class_name, prob in prob_items:
                        bar_length = int(prob * 30)
                        bar = "█" * bar_length + "░" * (30 - bar_length)
                        icon = "🔇" if class_name == "Silence" else "🔊"
                        print(f"   {icon} {class_name:<15}: {prob:.3f} |{bar}|")

                # Firebase status
                if "firebase_id" in result:
                    print_colored(
                        f"✅ Pushed to Firebase (ID: {result['firebase_id']})",
                        Colors.OKGREEN,
                    )
                elif "firebase_error" in result:
                    print_colored(
                        f"⚠️  Firebase: {result['firebase_error']}", Colors.WARNING
                    )

                print("-" * 80)

    except KeyboardInterrupt:
        print_colored("\n\n⏹️  Stopped real-time testing", Colors.OKCYAN)


def main():
    parser = argparse.ArgumentParser(
        description="Test client for audio classification server"
    )
    parser.add_argument(
        "--server",
        type=str,
        default=SERVER_URL,
        help=f"Server URL (default: {SERVER_URL})",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device ID (default: system default)",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=DEVICE_ID,
        help=f"Device identifier string (default: {DEVICE_ID})",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip VAD calibration (if already calibrated)",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip microphone test")

    args = parser.parse_args()

    print("=" * 80)
    print_colored("🎙️  AUDIO CLASSIFICATION TEST CLIENT", Colors.HEADER)
    print("=" * 80)

    # Check server health
    print("\n1. Checking server health...")
    server_ok, health_data = check_server_health(args.server)
    if not server_ok:
        print_colored("\n❌ Cannot proceed without server connection", Colors.FAIL)
        return

    # List audio devices
    print("\n2. Listing audio devices...")
    input_devices = list_audio_devices()

    # Choose device
    device_id = args.device
    if device_id is None:
        device_id = sd.default.device[0]
        print(f"\n   Using default device: {device_id}")
    elif device_id not in input_devices:
        print_colored(
            f"\n   ⚠️  Warning: Device {device_id} may not be a valid input device",
            Colors.WARNING,
        )

    # Test microphone
    if not args.skip_test:
        print("\n3. Testing microphone...")
        mic_ok = test_microphone(device_id=device_id)
        if not mic_ok:
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != "y":
                return
    else:
        print("\n3. Skipping microphone test")

    # Calibrate VAD
    if not args.skip_calibration:
        if health_data and not health_data.get("vad_calibrated", False):
            print("\n4. Calibrating VAD...")
            calibration_ok = calibrate_vad(args.server, device_id=device_id)
            if not calibration_ok:
                print_colored("\n   ⚠️  VAD calibration may have failed", Colors.WARNING)
                response = input("   Continue anyway? (y/n): ")
                if response.lower() != "y":
                    return
        else:
            print("\n4. VAD already calibrated")
    else:
        print("\n4. Skipping VAD calibration")

    # Start real-time testing
    print("\n5. Starting real-time classification...")
    start_real_time_testing(
        args.server, device_id_audio=device_id, device_id_str=args.device_id
    )


if __name__ == "__main__":
    main()
