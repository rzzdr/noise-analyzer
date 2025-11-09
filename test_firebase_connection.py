"""
Test Firebase Connection
Quick script to verify Firebase credentials are working
"""

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import sys


def test_firebase_connection():
    """Test Firebase connection and write a test document"""

    print("=" * 70)
    print("🔥 FIREBASE CONNECTION TEST")
    print("=" * 70)

    # Check if credentials file exists
    cred_file = "firebase-credentials.json"
    if not os.path.exists(cred_file):
        print(f"\n❌ Error: {cred_file} not found!")
        print("\nPlease create the credentials file:")
        print(
            "1. Go to: https://console.firebase.google.com/project/hawties-2a013/settings/serviceaccounts/adminsdk"
        )
        print("2. Click 'Generate new private key'")
        print(f"3. Save as '{cred_file}' in this directory")
        return False

    print(f"\n✅ Found credentials file: {cred_file}")

    # Initialize Firebase
    try:
        print("\n📡 Initializing Firebase Admin SDK...")
        cred = credentials.Certificate(cred_file)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin SDK initialized")
    except Exception as e:
        print(f"\n❌ Failed to initialize Firebase: {e}")
        return False

    # Get Firestore client
    try:
        print("\n📊 Connecting to Firestore...")
        db = firestore.client()
        print("✅ Connected to Firestore")
    except Exception as e:
        print(f"\n❌ Failed to connect to Firestore: {e}")
        return False

    # Test write operation
    try:
        print("\n✍️  Testing write operation...")
        test_data = {
            "test": True,
            "timestamp": datetime.now(),
            "message": "Firebase connection test",
            "source": "test_firebase_connection.py",
        }

        doc_ref = db.collection("test_connection").add(test_data)
        doc_id = doc_ref[1].id
        print(f"✅ Successfully wrote test document (ID: {doc_id})")
    except Exception as e:
        print(f"\n❌ Failed to write to Firestore: {e}")
        return False

    # Test read operation
    try:
        print("\n📖 Testing read operation...")
        docs = db.collection("test_connection").limit(5).stream()
        count = 0
        for doc in docs:
            count += 1
            doc_data = doc.to_dict()
            print(f"   Document {count}: {doc.id}")
            print(f"      Timestamp: {doc_data.get('timestamp', 'N/A')}")
            print(f"      Message: {doc_data.get('message', 'N/A')}")

        if count > 0:
            print(f"\n✅ Successfully read {count} test document(s)")
        else:
            print("\n⚠️  No documents found (this is OK for first run)")
    except Exception as e:
        print(f"\n❌ Failed to read from Firestore: {e}")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nFirebase is configured correctly and ready to use.")
    print("\nNext steps:")
    print("1. Start the Flask server: python flask_server.py")
    print("2. Run the test client: python test_client.py")
    print("\nYou can view the test data in Firebase Console:")
    print("https://console.firebase.google.com/project/hawties-2a013/firestore")
    print("=" * 70)

    return True


def cleanup_test_data():
    """Optional: Delete test documents"""
    response = input("\nDo you want to delete the test documents? (y/n): ")
    if response.lower() != "y":
        return

    try:
        db = firestore.client()
        docs = db.collection("test_connection").stream()
        count = 0
        for doc in docs:
            doc.reference.delete()
            count += 1

        if count > 0:
            print(f"\n✅ Deleted {count} test document(s)")
        else:
            print("\n⚠️  No test documents to delete")
    except Exception as e:
        print(f"\n❌ Error deleting test documents: {e}")


if __name__ == "__main__":
    success = test_firebase_connection()

    if success:
        cleanup_test_data()
    else:
        print("\n" + "=" * 70)
        print("❌ FIREBASE CONNECTION FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above and try again.")
        sys.exit(1)
