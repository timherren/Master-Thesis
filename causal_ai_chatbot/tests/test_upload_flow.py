#!/usr/bin/env python3
"""
Test script to verify upload and chat flow
Run this to test if data upload and retrieval works
"""

import requests
import json
import pandas as pd
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_upload_and_chat():
    """Test the full flow: upload data, then send a message"""
    
    print("=" * 60)
    print("Testing Upload and Chat Flow")
    print("=" * 60)
    print()
    
    # Create a test CSV file
    test_data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 3, 4, 5, 6],
        'x3': [3, 4, 5, 6, 7]
    })
    
    test_file = Path("test_data.csv")
    test_data.to_csv(test_file, index=False)
    print(f"✓ Created test file: {test_file}")
    print(f"  Shape: {test_data.shape}")
    print(f"  Columns: {list(test_data.columns)}")
    print()
    
    # Test 1: Upload data
    print("1. Testing data upload...")
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/api/upload_data", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Upload successful!")
            print(f"  Session ID: {data.get('session_id')}")
            print(f"  Status: {data.get('status')}")
            print(f"  Data info: {data.get('data_info')}")
            session_id = data.get('session_id')
        else:
            print(f"✗ Upload failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return
    
    print()
    
    # Test 2: Check session state
    print("2. Checking session state...")
    try:
        response = requests.get(f"{BASE_URL}/api/debug_session?session_id={session_id}")
        if response.status_code == 200:
            debug_data = response.json()
            print(f"✓ Session debug info:")
            print(json.dumps(debug_data, indent=2))
        else:
            print(f"✗ Debug check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Debug check error: {e}")
    
    print()
    
    # Test 3: Send a chat message
    print("3. Testing chat message...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "session_id": session_id,
                "message": "What is the effect of x1 on x3?"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Chat message sent!")
            print(f"  Session ID: {data.get('session_id')}")
            print(f"  Current step: {data.get('current_step')}")
            print(f"  Response preview: {data.get('response', '')[:200]}...")
        else:
            print(f"✗ Chat failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Chat error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    # Cleanup
    if test_file.exists():
        test_file.unlink()
        print(f"✓ Cleaned up test file")

if __name__ == "__main__":
    print("Make sure the chatbot server is running on http://localhost:8000")
    print("Press Enter to continue...")
    input()
    test_upload_and_chat()
