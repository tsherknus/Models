import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

from keras.src.layers import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

input = []
output = []

# TODO: Create a new model that attempts to predict % change over the following 5 days
# TODO: Get more training data, all rut 2000 data ideally.

def create_tensors(df, window):
    df_to_np = df.to_numpy()
    # Sliding window on the data frame
    for i in range(len(df_to_np)-window):
        forward_pass = []

        for x in df_to_np[i:i + window]:
            forward_pass.append([x])

        input.append(forward_pass)
        output.append(df_to_np[i+window])

# Specify the directory path
directory_path = 'DailyStockPriceData/'

# Loop through the directory and read all files
for filename in os.listdir(directory_path):
    print(filename)
    file_path = os.path.join(directory_path, filename)

    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)

        # data[:250].plot(x='Date', y='Close')
        # plt.show()

        create_tensors(data['Close'], 10)

input = np.array(input)
output = np.array(output)

print(input.shape)
print(output.shape)

training_passes = math.floor(len(input) * 0.8)
validation_test_passes = math.floor((len(input) - training_passes) / 2)

input_train = input[:training_passes]
output_train = output[:training_passes]
input_validation = input[training_passes:training_passes+validation_test_passes]
output_validation = output[training_passes:training_passes+validation_test_passes]
input_test = input[training_passes+validation_test_passes:]
output_test = output[training_passes+validation_test_passes:]

print(input.shape)
print(input_train.shape, output_train.shape)
print(input_validation.shape, output_validation.shape)
print(input_test.shape, output_test.shape)


model1 = Sequential()
model1.add(InputLayer((10, 1)))
model1.add(LSTM(128), dropout=0.2)
model1.add(Dense(64, 'relu'))
model1.add(Dropout(0.2))
model1.add(Dense(1, 'linear'))

model1.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
cp1 = ModelCheckpoint('model1/test.keras', save_best_only=True, verbose=1)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
model1.fit(input_train, output_train, validation_data=(input_validation, output_validation), epochs=50, callbacks=[early_stopping, cp1])

model1 = load_model('model1/test.keras')

test_predictions = model1.predict(input_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':output_test})

print(test_results[len(test_results)-50:])
test_results[len(test_results)-50:].plot()
plt.show()