import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Parameters
look_back = 60
parent_dir = '/content/drive/My Drive/Stock_Prediction_Project'

# Create directories for storing models and outputs
def create_directories(ticker):
    ticker_dir = os.path.join(parent_dir, ticker)
    model_dir = os.path.join(ticker_dir, 'model')
    output_dir = os.path.join(ticker_dir, 'output')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return model_dir, output_dir

# Define RSI computation
def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Define Bollinger Bands computation
def compute_bollinger_bands(series, window=20):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

# Define MACD computation
def compute_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Fetch and prepare data
def fetch_data(ticker='AMZN', start_date='2019-01-01'):
    data = yf.download(ticker, start=start_date)
    data['Close'] = data['Adj Close'].dropna()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['UpperBB'], data['LowerBB'] = compute_bollinger_bands(data['Close'])
    data['MACD'], data['Signal'] = compute_macd(data['Close'])
    data['VIX'] = yf.download('^VIX', start=start_date)['Adj Close']  # Volatility index
    data['VIX'] = data['VIX'].reindex(data.index, method='ffill').fillna(method='bfill')
    data.dropna(inplace=True)
    return data

# Create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and save the model
def train_and_save_model():
    # Prompt for ticker symbol
    ticker = input("Enter the ticker symbol of the stock you want to predict: ").upper()
    model_dir, output_dir = create_directories(ticker)

    # Fetch and prepare data
    data = fetch_data(ticker)
    features = ['Close', 'MA20', 'MA50', 'RSI', 'UpperBB', 'LowerBB', 'MACD', 'Signal', 'VIX']
    scaler = RobustScaler()
    data[features] = scaler.fit_transform(data[features])
    dataset = data[features].values

    # Prepare dataset
    def create_dataset(dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back)])
            Y.append(dataset[i + look_back, 0])  # Only predict the 'Close' price
        return np.array(X), np.array(Y)

    X, Y = create_dataset(dataset, look_back)
    X = X.reshape((X.shape[0], look_back, dataset.shape[1]))

    # Split data into training and test sets
    split = int(0.8 * X.shape[0])
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Create and train model
    model = create_lstm_model((look_back, X.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    previous_val_loss = np.inf
    no_improvement_count = 0
    max_no_improvement_epochs = 10

    while no_improvement_count < max_no_improvement_epochs:
        history = model.fit(X_train, Y_train, epochs=1, batch_size=32, validation_data=(X_test, Y_test), callbacks=[early_stopping])
        current_val_loss = history.history['val_loss'][-1]
        if current_val_loss < previous_val_loss:
            no_improvement_count = 0
            previous_val_loss = current_val_loss
        else:
            no_improvement_count += 1

    # Save the model
    model_path = os.path.join(model_dir, f'{ticker}_model.h5')
    model.save(model_path)
    print(f"Model for {ticker} saved at {model_path}.")

    # Predict on the test set and plot true vs. predicted prices
    Y_pred = model.predict(X_test)
    Y_pred = scaler.inverse_transform(np.concatenate([Y_pred, np.zeros((Y_pred.shape[0], len(features) - 1))], axis=1))[:, 0]
    Y_test_inverse = scaler.inverse_transform(np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], len(features) - 1))], axis=1))[:, 0]

    # Align the test dates and predictions
    test_dates = data.index[split+look_back:split+look_back+len(Y_test_inverse)]

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, Y_test_inverse, label='True Prices')
    plt.plot(test_dates, Y_pred, label='Predicted Prices')
    plt.title(f'{ticker} True vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{ticker}_True_vs_Predicted.png'))
    plt.close()

    print(f"True vs Predicted plot saved at {output_dir}.")

    # Create and save the true vs predicted table
    true_vs_pred_table = pd.DataFrame({'Date': test_dates, 'True Price': Y_test_inverse, 'Predicted Price': Y_pred})
    true_vs_pred_table['Difference'] = true_vs_pred_table['True Price'] - true_vs_pred_table['Predicted Price']
    table_csv_path = os.path.join(output_dir, f'{ticker}_True_vs_Predicted_Table.csv')
    true_vs_pred_table.to_csv(table_csv_path, index=False)
    print(f"True vs Predicted table saved at {table_csv_path}.")

# Example usage
train_and_save_model()
