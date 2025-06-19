import re
import tldextract
import joblib
import os
import urllib.parse
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load model
model_path = "model/phishing_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None  # If model not trained yet

# Suspicious keywords
keywords = ['login', 'secure', 'update', 'verify', 'account', 'bank', 'password', 'paypal']

# Feature extractor
def extract_features(url):
    features = []

    # Feature 1: URL length
    features.append(len(url))

    # Feature 2: @ symbol
    features.append(1 if '@' in url else 0)

    # Feature 3: IP address
    features.append(1 if re.match(r"http[s]?://(?:\d{1,3}\.){3}\d{1,3}", url) else 0)

    # Feature 4: HTTPS used
    features.append(1 if url.startswith("https") else 0)

    # Feature 5: Dot count
    features.append(url.count('.'))

    # Feature 6: Suspicious keywords
    features.append(1 if any(k in url.lower() for k in keywords) else 0)

    return np.array(features).reshape(1, -1)

# ML prediction function
def predict_phishing(url):
    if model:
        features = extract_features(url)
        prediction = model.predict(features)[0]
        return prediction == 1  # 1 = Phishing
    else:
        return None

# Main function
if __name__ == "__main__":
    url = input("Enter the URL to scan: ")

    # ML prediction (if model is available)
    result = predict_phishing(url)

    if result is None:
        print("⚠️ No trained ML model found. Run training script first.")
    elif result:
        print("❌ Phishing URL Detected (ML Model)")
    else:
        print("✅ URL looks safe (ML Model)")
