"""
Simple test to check if forecast endpoint works
"""
import requests
import json

# Test 1: Check if there's training data
print("=" * 50)
print("Test 1: Checking training data")
print("=" * 50)

try:
    response = requests.get("http://localhost:5000/training-data?product_id=1")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} training records")
        if len(data) > 0:
            print(f"First record: {data[0]}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Try forecast
print("\n" + "=" * 50)
print("Test 2: Testing forecast endpoint")
print("=" * 50)

try:
    response = requests.post(
        "http://localhost:5000/forecast-inputs/1",
        headers={"Content-Type": "application/json"},
        data=json.dumps({})
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
