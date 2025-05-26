# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('models/voice.csv')
X = data.drop(['label'], axis=1)
y = data['label'].map({'male':0, 'female':1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(30,), max_iter=500)
model.fit(X_train, y_train)

joblib.dump(model, 'models/gender_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')