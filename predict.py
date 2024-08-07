import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Prompt for ticker symbol
ticker = input("Enter the ticker symbol of the stock you want to predict: ").upper()

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

# Function to fetch data from Yahoo Finance
def fetch_data(ticker='AAPL', start_date='2019-01-01'):
    data = yf.download(ticker, start=start_date)
    data['Close'] = data['Adj Close'].dropna()
    data['Volume'] = data['Volume']
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['UpperBB'], data['LowerBB'] = compute_bollinger_bands(data['Close'])
    data['MACD'], data['Signal'] = compute_macd(data['Close'])
    data['VIX'] = yf.download('^VIX', start=start_date)['Adj Close']  # Volatility index
    data['VIX'] = data['VIX'].reindex(data.index, method='ffill').fillna(method='bfill')
    data.dropna(inplace=True)
    return data

# Function to prepare dataset
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Directory paths
parent_dir = '/content/drive/My Drive/Stock_Prediction_Project'
model_dir = os.path.join(parent_dir, ticker, 'model')
plot_dir = os.path.join(parent_dir, ticker, 'output')
table_dir = os.path.join(parent_dir, ticker, 'output')

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

# File paths
model_path = os.path.join(model_dir, f'{ticker}_model.h5')
plot_path = os.path.join(plot_dir, f'{ticker}_Prediction_Plot.png')
table_csv_path = os.path.join(table_dir, f'{ticker}_Prediction_Table.csv')

# Load the model
model = load_model(model_path)

# Fetch and prepare data
data = fetch_data(ticker)
features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'UpperBB', 'LowerBB', 'MACD', 'Signal', 'VIX']
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data[features])
dataset = scaled_data

# Print dataset shape
print(f'Dataset shape: {dataset.shape}')

# Split data
look_back = 60
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

# Ensure the dataset splits properly
print(f'Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}')

X_train, Y_train = create_dataset(dataset[:train_size], look_back)
X_val, Y_val = create_dataset(dataset[train_size:train_size+val_size], look_back)
X_test, Y_test = create_dataset(dataset[train_size+val_size:], look_back)

# Print shapes before reshaping
print(f'X_test shape before reshaping: {X_test.shape}')
print(f'Y_test shape: {Y_test.shape}')

# Reshape X_test
X_test = X_test.reshape((X_test.shape[0], look_back, dataset.shape[1]))

# Print shapes after reshaping
print(f'X_test shape after reshaping: {X_test.shape}')

# Generate test dates
test_dates = data.index[train_size+val_size+look_back:train_size+val_size+look_back + len(Y_test)]

# Function to predict future prices based on the model
def predict_next_days(model, last_window, days=30):
    future_prices = []
    current_batch = last_window.reshape((1, look_back, last_window.shape[1]))
    for _ in range(days):
        predicted_price = model.predict(current_batch)[0][0]
        future_prices.append(predicted_price)  # Keep the scale as is
        predicted_features = np.array([[predicted_price] + [0] * (last_window.shape[1] - 1)])  # Fill with zeros for other features
        current_batch = np.append(current_batch[:, 1:, :], predicted_features.reshape(1, 1, last_window.shape[1]), axis=1)
    # Expand future_prices to match the number of features (10 in this case)
    future_prices_expanded = np.array(future_prices).reshape(-1, 1)
    future_prices_expanded = np.concatenate([future_prices_expanded] + [np.zeros_like(future_prices_expanded) for _ in range(9)], axis=1)
    future_prices = scaler.inverse_transform(future_prices_expanded)[:, 0]  # Only get the closing prices
    return future_prices

# Function to save predictions and outputs
def save_predictions_and_outputs():
    last_window = dataset[-look_back:]
    print(f'Last window for predictions:\n{last_window}')
    
    predicted_prices = predict_next_days(model, last_window)

    print(f'Predicted prices (before adjustment):\n{predicted_prices}')

    last_date = data.index[-1]
    future_dates = pd.date_range(last_date, periods=30, freq='B').tolist()
    previous_month = data['Close'][-44:]  # Approximate number of trading days in 2 months

    print(f'Previous month closing prices:\n{previous_month.values}')
    
    adjusted_predicted_prices = predicted_prices  # Already adjusted by inverse transform

    print(f'Adjusted predicted prices:\n{adjusted_predicted_prices}')

    plt.figure(figsize=(12, 8))
    plt.plot(previous_month.index, previous_month.values, color='green', label='True Prices')
    plt.plot(future_dates, adjusted_predicted_prices, color='blue', label='Predicted Prices')

    week_indices = [4, 9, 14, 19, 24, 29]
    for i in week_indices:
        plt.annotate(f'{adjusted_predicted_prices[i]:.2f}',
                     (future_dates[i], adjusted_predicted_prices[i]),
                     textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{future_dates[i].strftime("%Y-%m-%d")}',
                     (future_dates[i], adjusted_predicted_prices[i]),
                     textcoords="offset points", xytext=(0,-15), ha='center')
    plt.scatter([future_dates[i] for i in week_indices], [adjusted_predicted_prices[i] for i in week_indices], color='red')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

    # Prepare the table of combined true and predicted prices
    combined_table = pd.DataFrame({
        'Date': previous_month.index.tolist() + future_dates,
        'Price': np.concatenate((previous_month.values, adjusted_predicted_prices))
    }).round(2)  # Round numbers to 2 decimals
    print(combined_table)

    # Save the table as a CSV file
    combined_table.to_csv(table_csv_path, index=False)
    print(f"Prediction table saved as CSV at {table_csv_path}.")

    # Save the closing prices for the past month as a CSV file
    past_month_csv_path = os.path.join(table_dir, f'{ticker}_Past_Month_Closing_Prices.csv')
    previous_month.to_csv(past_month_csv_path, header=['Close'], index=True)
    print(f"Past month's closing prices saved as CSV at {past_month_csv_path}.")

# Function to create LSTM model with regularization
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Example usage of creating a new model with regularization
input_shape = (look_back, len(features))
model = create_lstm_model(input_shape)
model.summary()

# Save predictions and outputs
save_predictions_and_outputs()
