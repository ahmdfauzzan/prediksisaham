import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import random
import os

# Mulai pencatatan waktu eksekusi
start_time = time.time()

# Konfigurasi Streamlit
st.set_page_config(layout="wide")
st.title("Prediksi Harga Saham PT Astra International Tbk (ASII) Menggunakan ARIMA")

# Upload file CSV
data_file = st.file_uploader("Unggah file CSV", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Tampilkan informasi data
    st.subheader("Informasi Data")
    st.write("Jumlah baris dan kolom:", data.shape)
    st.write("\nDeskripsi Statistik Data:")
    st.write(data.describe())
    st.write("\nInformasi Data:")
    st.write(data.info())
    st.write("\nMissing Values:")
    st.write(data.isna().sum())
    st.write("\nData (5 baris pertama dan terakhir):")
    st.write(pd.concat([data.head(5), data.tail(5)]))
    
    # Visualisasi
    st.subheader("Visualisasi Data")
    selected_column = st.selectbox("Pilih kolom untuk visualisasi:", ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends'])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data[selected_column], label=selected_column, linewidth=2)
    ax.set_title(f'Pergerakan {selected_column} Saham ASII')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (IDR)')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Pra-pemrosesan data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = data.copy()
    data_scaled[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']])
    
    train_size = int(len(data_scaled) * 0.9)
    train_data, test_data = data_scaled.iloc[:train_size], data_scaled.iloc[train_size:]
    
    st.subheader("Pembagian Data")
    st.write(f"Jumlah data train: {len(train_data)}")
    st.write(f"Jumlah data test: {len(test_data)}")
    
    # Menerapkan auto_arima
    st.subheader("Model ARIMA")
    with st.spinner("Menjalankan auto_arima, harap tunggu..."):
        arima_model = auto_arima(train_data['Close'], start_p=0, start_q=0,
                                 max_p=5, max_q=5, max_d=2,
                                 seasonal=False, trace=True,
                                 error_action='ignore', suppress_warnings=True,
                                 stepwise=True)
    st.write(arima_model.summary())
    
    # Fit model ARIMA
    arima_best_model = ARIMA(train_data['Close'], order=arima_model.order)
    arima_fitted_model = arima_best_model.fit()
    
    # Evaluasi Model ARIMA
    test_predictions = arima_fitted_model.forecast(steps=len(test_data))
    test_predictions = np.array(test_predictions)
    
    mae_arima = mean_absolute_error(test_data['Close'], test_predictions)
    mse_arima = mean_squared_error(test_data['Close'], test_predictions)
    rmse_arima = np.sqrt(mse_arima)
    mape_arima = np.mean(np.abs((test_data['Close'] - test_predictions) / test_data['Close'])) * 100
    
    st.subheader("Evaluasi Model ARIMA")
    st.write(f"MAE: {mae_arima:.4f}")
    st.write(f"MSE: {mse_arima:.4f}")
    st.write(f"RMSE: {rmse_arima:.4f}")
    st.write(f"MAPE: {mape_arima:.2f}%")
    
    # Denormalisasi hasil prediksi
    test_data_denormalized = scaler.inverse_transform(test_data)
    test_predictions_denormalized = scaler.inverse_transform(np.concatenate((test_predictions.reshape(-1, 1), test_data.iloc[:, 1:].values), axis=1))[:, 0]
    
    # Evaluasi setelah denormalisasi
    mae_arima_denormalized = mean_absolute_error(test_data_denormalized[:, 3], test_predictions_denormalized)
    mse_arima_denormalized = mean_squared_error(test_data_denormalized[:, 3], test_predictions_denormalized)
    rmse_arima_denormalized = np.sqrt(mse_arima_denormalized)
    mape_arima_denormalized = np.mean(np.abs((test_data_denormalized[:, 3] - test_predictions_denormalized) / test_data_denormalized[:, 3])) * 100
    
    st.subheader("Evaluasi ARIMA Setelah Denormalisasi")
    st.write(f"MAE: {mae_arima_denormalized:.2f}")
    st.write(f"MSE: {mse_arima_denormalized:.2f}")
    st.write(f"RMSE: {rmse_arima_denormalized:.2f}")
    st.write(f"MAPE: {mape_arima_denormalized:.2f}%")
    
    # Visualisasi Prediksi
    st.subheader("Visualisasi Prediksi ARIMA")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_data.index, test_data_denormalized[:, 3], label='Actual Close Price', color='blue', linewidth=2)
    ax.plot(test_data.index, test_predictions_denormalized, label='Predicted Close Price (ARIMA)', color='red', linewidth=2)
    ax.set_title('ARIMA: Actual vs Predicted Close Price')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Penutupan (IDR)')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Hitung waktu eksekusi
    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"\nWaktu eksekusi seluruh skrip: {execution_time:.2f} detik")


st.title("Prediksi Harga Saham PT Astra International Tbk (ASII) Menggunakan LSTM")

# Fungsi untuk set seed
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)

# Upload file CSV
data_file = st.file_uploader("Upload file CSV", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    st.write("### Statistik Data")
    st.write(data.describe())
    
    # Preprocessing Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = data.copy()
    data_scaled[['Close']] = scaler.fit_transform(data[['Close']])
    
    train_size = int(len(data_scaled) * 0.9)
    train_data, test_data = data_scaled.iloc[:train_size], data_scaled.iloc[train_size:]
    
    # Fungsi untuk membuat dataset dengan look_back timestep
    def create_dataset(dataset, look_back=10):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    look_back = 10
    trainX, trainY = create_dataset(train_data[['Close']].values, look_back)
    testX, testY = create_dataset(test_data[['Close']].values, look_back)
    
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    # Model LSTM dengan parameter tuning
    def build_model(optimizer='adam', dropout_rate=0.2):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(trainX.shape[1], 1)),
            Dropout(dropout_rate),
            LSTM(50),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model
    
    # Grid Search Manual
    param_grid = {
        'batch_size': [16],
        'epochs': [100],
        'optimizer': ['adam'],
        'dropout_rate': [0.2]
    }
    
    best_model = None
    best_score = float('inf')
    best_params = None
    
    for params in ParameterGrid(param_grid):
        model = build_model(optimizer=params['optimizer'], dropout_rate=params['dropout_rate'])
        model.fit(trainX, trainY, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, shuffle=False)
        predictions = model.predict(testX)
        score = mean_squared_error(testY, predictions)
        
        if score < best_score:
            best_score = score
            best_model = model
            best_params = params
    
    st.write("### Parameter Model Terbaik")
    st.json(best_params)
    
    # Prediksi menggunakan model terbaik
    predictions = best_model.predict(testX)
    predictions_original = scaler.inverse_transform(predictions)
    testY_original = scaler.inverse_transform(testY.reshape(-1, 1))
    
    # Evaluasi
    def evaluate_predictions(true, pred):
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((true - pred) / true)) * 100
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
    
    eval_results = evaluate_predictions(testY_original, predictions_original)
    st.write("### Evaluasi Model Setelah Denormalisasi")
    st.json(eval_results)
    
    # Plot hasil prediksi
    plt.figure(figsize=(10, 5))
    time_index = data.index[len(train_data) + look_back:len(train_data) + look_back + len(testY_original)]
    plt.plot(time_index, testY_original, label='Actual Price', color='blue')
    plt.plot(time_index, predictions_original, label='Predicted Price (LSTM)', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (IDR)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
