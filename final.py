# General
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Audio processing
import librosa
import librosa.display

# ML & Deep Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



# Paths (CHANGE THESE)
DATA_DIR = 'Dataset\\audios'
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, 'train')
TEST_AUDIO_DIR = os.path.join(DATA_DIR, 'test')

# Load data
df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
df_train.head()

def extract_mfcc_features(file_path, sr=16000, n_mfcc=13):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1), mfcc.T
    except Exception as e:
        print(f'Failed for {file_path}: {e}')
        return np.zeros(n_mfcc), np.zeros((100, n_mfcc))
    
X_mfcc, X_seq, y = [], [], []
for fname, label in zip(df_train['filename'], df_train['label']):
    mean_feat, seq_feat = extract_mfcc_features(os.path.join(TRAIN_AUDIO_DIR, fname))
    X_mfcc.append(mean_feat)
    X_seq.append(seq_feat[:100])  # pad/truncate to 100 timesteps
    y.append(label)

X_mfcc = np.array(X_mfcc)
X_seq = np.array(X_seq)
y = np.array(y)
# XGBoost Model
X_train, X_val, y_train, y_val = train_test_split(X_mfcc, y, test_size=0.2, random_state=42)
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
rmse_xgb = mean_squared_error(y_val, y_pred_xgb, squared=False)
corr_xgb, _ = pearsonr(y_val, y_pred_xgb)
print(f'XGBoost RMSE: {rmse_xgb:.4f}, Pearson Corr: {corr_xgb:.4f}')

X_seq_train, X_seq_val, y_seq_train, y_seq_val = train_test_split(X_seq, y, test_size=0.2, random_state=42)

lstm_model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

lstm_model.fit(X_seq_train, y_seq_train, validation_data=(X_seq_val, y_seq_val),
                epochs=30, batch_size=16, callbacks=[early_stop], verbose=1)

y_pred_lstm = lstm_model.predict(X_seq_val).squeeze()
rmse_lstm = mean_squared_error(y_seq_val, y_pred_lstm, squared=False)
corr_lstm, _ = pearsonr(y_seq_val, y_pred_lstm)
print(f'LSTM RMSE: {rmse_lstm:.4f}, Pearson Corr: {corr_lstm:.4f}')

X_test = []
for fname in df_test['filename']:
    mean_feat, _ = extract_mfcc_features(os.path.join(TEST_AUDIO_DIR, fname))
    X_test.append(mean_feat)
X_test = np.array(X_test)

test_preds = xgb_model.predict(X_test)
sample_submission['label'] = test_preds
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()