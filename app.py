from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = joblib.load('models/gender_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def extract_features(file_path):
    X, sample_rate = librosa.load(file_path)
    
    # Рассчитываем все признаки как в оригинальном датасете
    features = {
        'meanfreq': np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate)),
        'sd': np.std(X),
        'median': np.median(X),
        'Q25': np.quantile(X, 0.25),
        'Q75': np.quantile(X, 0.75),
        'IQR': np.quantile(X, 0.75) - np.quantile(X, 0.25),
        'skew': librosa.feature.spectral_bandwidth(y=X, sr=sample_rate).mean(),
        'kurt': librosa.feature.spectral_contrast(y=X, sr=sample_rate).mean(),
        'sp.ent': librosa.feature.spectral_flatness(y=X).mean(),
        'sfm': librosa.feature.spectral_rolloff(y=X, sr=sample_rate).mean(),
        'mode': librosa.feature.zero_crossing_rate(X).mean(),
        'centroid': librosa.feature.spectral_centroid(y=X, sr=sample_rate).mean(),
        'meanfun': np.mean(librosa.effects.harmonic(X)),
        'minfun': np.min(librosa.effects.harmonic(X)),
        'maxfun': np.max(librosa.effects.harmonic(X)),
        'meandom': np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)),
        'mindom': np.min(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)),
        'maxdom': np.max(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)),
        'dfrange': np.max(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate) - 
                   np.min(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate))),
        'modindx': (np.max(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)) - 
                   np.min(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate))) / 
                   np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate))
    }
    
    return list(features.values())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return 'No file uploaded', 400
    
    audio = request.files['audio']
    if audio.filename == '':
        return 'No selected file', 400
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"audio_{timestamp}.wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio.save(filepath)
    
    try:
        features = extract_features(filepath)
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        result = 'Мужской' if prediction == 0 else 'Женский'
    except Exception as e:
        return f"Error processing audio: {str(e)}", 500
    finally:
        os.remove(filepath)
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=8000)