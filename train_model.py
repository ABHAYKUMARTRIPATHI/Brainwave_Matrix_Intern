import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np
import re

# Fake mini dataset (you can replace with real one)
data = {
    'url': [
        'https://paypal.verify-login.com',
        'https://google.com',
        'http://192.168.1.1/login',
        'https://secure-bank-update.com',
        'http://example.com',
    ],
    'label': [1, 0, 1, 1, 0]  # 1 = phishing, 0 = safe
}

df = pd.DataFrame(data)

def extract_features(url):
    return [
        len(url),
        1 if '@' in url else 0,
        1 if re.match(r"http[s]?://(?:\d{1,3}\.){3}\d{1,3}", url) else 0,
        1 if url.startswith("https") else 0,
        url.count('.'),
        1 if any(k in url.lower() for k in ['login', 'secure', 'update', 'verify', 'account', 'bank', 'password', 'paypal']) else 0,
    ]

# Feature extraction
df['features'] = df['url'].apply(extract_features)
X = np.array(df['features'].tolist())
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/phishing_model.pkl")
print("✅ Model trained and saved.")
