import re
import tldextract
import joblib
import os
import urllib.parse
from sklearn.ensemble import RandomForestClassifier
import numpy as np
model_path = "model/phishing_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None  
keywords = ['login', 'secure', 'update', 'verify', 'account', 'bank', 'password', 'paypal']
def extract_features(url):
    features = []
    features.append(len(url))
    features.append(1 if '@' in url else 0)
    features.append(1 if re.match(r"http[s]?://(?:\d{1,3}\.){3}\d{1,3}", url) else 0)
    features.append(1 if url.startswith("https") else 0)
    features.append(url.count('.'))
    features.append(1 if any(k in url.lower() for k in keywords) else 0)
    return np.array(features).reshape(1, -1)    
def predict_phishing(url):
    if model:
        features = extract_features(url)
        prediction = model.predict(features)[0]
        return prediction == 1  # 1 = Phishing
    else:
        return None
if __name__ == "__main__":
    url = input("Enter the URL to scan: ")
    result = predict_phishing(url)
    if result is None:
        print("⚠️ No trained ML model found. Run training script first.")
    elif result:
        print("❌ Phishing URL Detected (ML Model)")
    else:
        print("✅ URL looks safe (ML Model)")
