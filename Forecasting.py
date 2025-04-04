# Required Library list
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Tensorflow log adjustment to show only errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load dataset that contains 2 columns "Date" and "Liftings"
data = pd.read_excel("Liftings2.xlsx")
data['Date'] = pd.to_datetime(data['Date'])

# Specifying period in which to extract data and use for the model. 
prediction = data.loc[
    (data['Date'] > datetime(2022,1,1)) &
    (data['Date'] < datetime(2024,11,1))
]

# Extracting date and lifting from dataset.
Date = data.filter(["Date"]) 
Lifted = data.filter(["Liftings"])
# Converting to numpy array
dataset = Lifted.values 
# Defining traning data length at 95 % of dataset.
Training_data_len = int(np.ceil(len(dataset)*0.95))
# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)
# Extracting the traning data
Training_data = scaled_data[:Training_data_len] #95% of all data

# Training sequences 
X_train, Y_train = [], []
for i in range(30, len(Training_data)): # 30 day sequence
    X_train.append(Training_data[i-30:i, 0]) # 30 previous values as input
    Y_train.append(Training_data[i,0]) # Next day target

# Converting training data to fit LSTM model
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# LSTM Model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.LSTM(64, return_sequences=False)) 
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary()

# Compiling the model with Adam optimizer and Mean Squared Error loss function
model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])

# Training the model
training = model.fit(X_train, Y_train, epochs=10, batch_size=8)

# Prepare test data (last 30 days of training data + test data)
test_data = scaled_data[Training_data_len-30:]
X_test, y_test = [],dataset[Training_data_len:]
for i in range (30, len(test_data)):
    X_test.append(test_data[i-30:i, 0]) # Sequence for testing

# Converting test data to fit LSTM model
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

# Creating predictions and inverse to original scale.
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Spliting original dataset as training and test sets
train = data[:Training_data_len]
test = data[Training_data_len:] 
# Creating a copy to avoid modifying original data
test = test.copy()
# Updating predictions to test data
test['Predictions'] = predictions

# Function to generate multi-step forecasts
def multi_step_forecasts(n_past, n_future):
    x = X_test  # Using existing test data
    y = predictions.reshape(-1, 1)  # Using existing predictions
    
     # Extracting last observed sequence
    x_past = x[- n_past - 1:, :, :][:1]  # last observed input sequence
    y_past = y[- n_past - 1]             # last observed target value
    y_future = []                        # storing future predictions

    # Generating future forecasts
    for i in range(n_past + n_future):
        x_past = np.append(x_past[:, 1:, :], y_past.reshape(1, 1, 1), axis=1) #Shifting sequence 
        y_past = model.predict(x_past) # Predicting next
        y_future.append(y_past.flatten()[0]) # Saving prediction

    # Transforming the forecasts back to the original scale
    y_future = scaler.inverse_transform(np.array(y_future).reshape(-1, 1)).flatten()

    # Generating future dates starting from last date in test data
    last_date = test['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future)
    
    # Creating DataFrame for future predictions
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Future_Predictions': y_future
    })
    
    return future_df

# Generating future forecasts for the next 90 days
future_forecasts = multi_step_forecasts(n_past=0, n_future=90)

# Create a combined DataFrame with test data that already has actuals and predictions
combined_df = test.copy()

# Combining test actuals, test predictions, and future predictions
combined_df = pd.concat([combined_df, future_forecasts], sort=True)

# Saving combined results to Excel
combined_df.to_excel("Combined_Forecasts.xlsx", index=False)

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(train['Date'], train['Liftings'], label="Training Data", color="gray", alpha=0.5)
plt.plot(test['Date'], test['Liftings'], label="Test Actuals", color="blue")
plt.plot(test['Date'], test['Predictions'], label="Test Predictions", color="red")
plt.plot(future_forecasts['Date'], future_forecasts['Future_Predictions'], label="Future Forecasts", color="green", linestyle='--')
plt.xlabel("Date")
plt.ylabel("Liftings")
plt.title("Liftings Forecast (Historical + Future)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Combined_Forecast_Plot.png", dpi=300)
plt.show()
