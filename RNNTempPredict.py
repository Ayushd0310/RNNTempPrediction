import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Load the temperature data
data = pd.read_csv('C:/Users/91967/OneDrive/Desktop/Machinelearning/Data-2024passout.csv', index_col='YEAR', parse_dates=True)
temperature = data['TEMP(C)'].values.reshape(-1, 1)

# Scale the temperature data to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
temperature = scaler.fit_transform(temperature)

# Create input-output sequences with a window size of 7 days
window_size = 7
input_data = []
output_data = []
for i in range(len(temperature) - window_size):
    input_data.append(temperature[i:i+window_size])
    output_data.append(temperature[i+window_size])
input_data = np.array(input_data)
output_data = np.array(output_data)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(input_data))
train_input = input_data[:split_index]
train_output = output_data[:split_index]
test_input = input_data[split_index:]
test_output = output_data[split_index:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(train_input, train_output, epochs=100, batch_size=32, validation_data=(test_input, test_output))

# Plot the training and testing loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the testing set
test_loss = model.evaluate(test_input, test_output)
print('Test loss:', test_loss)

# Make predictions on new data
new_data = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
new_data = new_data.reshape(1, window_size, 1)
prediction = model.predict(new_data)
prediction = scaler.inverse_transform(prediction)
print('Next temperature prediction:', prediction)

# Plot the actual temperature values and the predicted values
actual_values = scaler.inverse_transform(test_output.reshape(-1, 1))
predicted_values = scaler.inverse_transform(model.predict(test_input))
plt.plot(actual_values, label='Actual Temperature')
plt.plot(predicted_values, label='Predicted Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()
