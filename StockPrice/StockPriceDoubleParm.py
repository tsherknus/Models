import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from keras.src.layers import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

data = pd.read_csv('/Users/sherknus-family/PycharmProjects/DataAgregator/DailyStockPriceData/TSLA.csv')

# data[:250].plot(x='Date', y='Close')
# plt.show()

def create_tensors(df, window):
    close_to_np = df['Close'].to_numpy()
    volume_to_np = df['Open'].to_numpy()
    input = []
    output = []
    # Sliding window on the data frame
    for i in range(len(close_to_np)-window):
        forward_pass = []

        for y in range(window):
            forward_pass.append([close_to_np[i+y], volume_to_np[i+y]])

        # for x in close_to_np[i:i + window]:
        #     forward_pass.append([x])

        input.append(forward_pass)
        output.append(close_to_np[i+window])

    return np.array(input), np.array(output)

new_df = pd.DataFrame(columns=['Close', 'Volume'])

new_df['Close'] = data['Close']
new_df['Open'] = data['Open']


input, output = create_tensors(new_df, 50)

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
model1.add(InputLayer((50, 2)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

cp1 = ModelCheckpoint('model1/test.keras', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model1.fit(input_train, output_train, validation_data=(input_validation, output_validation), epochs=100, callbacks=[cp1])

model1 = load_model('multiparam/test.keras')

test_predictions = model1.predict(input_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':output_test})
print(test_results)