import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------
# 1. LOAD DATA FROM CSV FILE
# --------------------------------------
def load_csv_file(csv_path):
    """
    Loads a CSV file into a DataFrame.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame created from the CSV file.
    """
    df = pd.read_csv(csv_path)
    print("CSV file loaded from:", csv_path)
    return df


# --------------------------------------
# 2. DATA PREPROCESSING FUNCTIONS
# --------------------------------------
def preprocess_data(df, feature_column="soil_temperature", date_column="date"):
    """
    Sorts data by date (if a date column is present) and applies MinMax scaling to the target feature.

    Parameters:
        df (pd.DataFrame): The input data.
        feature_column (str): Name of the target feature column.
        date_column (str): Name of the date column (if exists).

    Returns:
        tuple: Scaled data as a numpy array and the scaler object.
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=date_column).reset_index(drop=True)
    data = df[[feature_column]].values.astype("float32")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def create_sequences(data, look_back=30, forecast_steps=6):
    """
    Converts a continuous time series into input/output sequences for supervised learning.

    Parameters:
        data (np.array): Scaled time series data.
        look_back (int): Number of past time steps used as input.
        forecast_steps (int): Number of future time steps to predict.

    Returns:
        tuple: Arrays for inputs (X) and targets (y). X is reshaped for LSTM use.
    """
    X, y = [], []
    for i in range(len(data) - look_back - forecast_steps + 1):
        seq_X = data[i:(i + look_back), 0]
        seq_y = data[(i + look_back):(i + look_back + forecast_steps), 0]
        X.append(seq_X)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


# --------------------------------------
# 3. MODEL DEFINITIONS
# --------------------------------------
def build_lstm_model(input_shape, forecast_steps):
    """
    Builds and compiles a unidirectional LSTM model.

    Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features).
        forecast_steps (int): Number of forecast steps as output neurons.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='tanh', return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_steps))
    model.compile(optimizer='adam', loss='mse')
    print("LSTM model summary:")
    model.summary()
    return model


def build_bilstm_model(input_shape, forecast_steps):
    """
    Builds and compiles a bidirectional LSTM (BiLSTM) model.

    Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features).
        forecast_steps (int): Number of forecast steps as output neurons.

    Returns:
        tf.keras.Model: Compiled BiLSTM model.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='tanh', return_sequences=False), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_steps))
    model.compile(optimizer='adam', loss='mse')
    print("BiLSTM model summary:")
    model.summary()
    return model


# --------------------------------------
# 4. TRAINING AND EVALUATION FUNCTION
# --------------------------------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler, forecast_steps, epochs=100, batch_size=32):
    """
    Trains the given model using the training set and evaluates its performance on the test set.

    Parameters:
        model (tf.keras.Model): Model to be trained.
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        scaler (MinMaxScaler): Scaler used to invert scaling.
        forecast_steps (int): Number of forecast steps.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        tuple: Training history and average R² score over the forecast horizon.
    """
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    y_pred = model.predict(X_test)
    y_test_rescaled = scaler.inverse_transform(y_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    r2_all = []
    mse_all = []
    for i in range(forecast_steps):
        r2 = r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
        mse = mean_squared_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
        r2_all.append(r2)
        mse_all.append(mse)
        print(f"Forecast Step {i + 1}: R2 = {r2:.4f}, MSE = {mse:.4f}")

    avg_r2 = np.mean(r2_all)
    print(f"Average R2 score over forecast horizon: {avg_r2:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(y_test_rescaled[0], marker='o', label="Actual")
    plt.plot(y_pred_rescaled[0], marker='x', label="Predicted")
    plt.title("Soil Temperature Forecast - First Test Sample")
    plt.xlabel("Forecast Step (Month)")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

    return history, avg_r2


# --------------------------------------
# 5. MAIN EXECUTION PIPELINE
# --------------------------------------
def main():
    # Path to the CSV file containing the data.
    csv_file_path = "/Users/rubenmunoz/Downloads/psspredict_new01.csv"

     # Load the DataFrame from the CSV file.
    df = load_csv_file(csv_file_path)
    print("Columns in CSV:", df.columns)

    # Use an existing column (e.g., "naturaltemperature_5") as the target feature.
    scaled_data, scaler = preprocess_data(df, feature_column="naturaltemperature_5", date_column="date")

    # Set forecast and sequence parameters.
    forecast_steps = 6  # Forecasting the next 6 time steps (months)
    look_back = 30  # Use the previous 30 time steps (e.g., days) as input

    # Create input and output sequences for the model.
    X, y = create_sequences(scaled_data, look_back=look_back, forecast_steps=forecast_steps)

    # Split the sequences into training and testing sets (80/20 split).
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Get the input shape for the models.
    input_shape = (X_train.shape[1], X_train.shape[2])

    # ---------------------
    # Train LSTM Model
     # ---------------------
    print("\n--- Training LSTM Model ---")
    lstm_model = build_lstm_model(input_shape, forecast_steps)
    history_lstm, avg_r2_lstm = train_and_evaluate(lstm_model, X_train, y_train, X_test, y_test, scaler,
                                                       forecast_steps)

    # ---------------------
    # Train BiLSTM Model
    # ---------------------
    print("\n--- Training BiLSTM Model ---")
    bilstm_model = build_bilstm_model(input_shape, forecast_steps)
    history_bilstm, avg_r2_bilstm = train_and_evaluate(bilstm_model, X_train, y_train, X_test, y_test, scaler,
                                                           forecast_steps)

    # Check if either model has reached the desired performance (average R2 >= 0.95)
    if avg_r2_lstm >= 0.95 or avg_r2_bilstm >= 0.95:
        print("\nTarget performance (>= 95% R2 score) has been achieved by one of the models.")
    else:
        print(
            "\nNeither model reached the target performance. Consider further tuning of hyperparameters or additional feature engineering.")
    # ---------------------
    # Train BiLSTM Model
    # ---------------------
    print("\n--- Training BiLSTM Model ---")
    bilstm_model = build_bilstm_model(input_shape, forecast_steps)
    history_bilstm, avg_r2_bilstm = train_and_evaluate(bilstm_model, X_train, y_train, X_test, y_test, scaler,
                                                       forecast_steps)

    # Check if either model has reached the desired performance (average R2 >= 0.95)
    if avg_r2_lstm >= 0.95 or avg_r2_bilstm >= 0.95:
        print("\nTarget performance (>= 95% R2 score) has been achieved by one of the models.")
    else:
        print(
            "\nNeither model reached the target performance. Consider further tuning of hyperparameters or additional feature engineering.")


if __name__ == "__main__":
    main()
