
import requests
import json
import time

def test_api():
    url = "http://localhost:8001/predict/risk"
    payload = {
        "gender": "Female",
        "age": 55.0,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "past",
        "bmi": 32.0,
        "HbA1c_level": 7.0,
        "blood_glucose_level": 200
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("Response:", json.dumps(data, indent=2))
            if "model" in data and "Grandmaster" in data["model"]:
                print("SUCCESS: API is serving the Grandmaster Model!")
            else:
                print("WARNING: API response format unexpected or model missing.")
        else:
            print(f"FAILED: Status Code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Ensure the API server is running on port 8001.")

if __name__ == "__main__":
    # Wait a sec for server startup
    time.sleep(5)
    test_api()
