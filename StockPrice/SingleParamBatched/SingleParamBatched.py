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

# TODO: Create a new model that attempts to predict % change over the following 5 days
# TODO: Get more training data, all rut 2000 data ideally.

directory_path = '../DailyStockPriceData/'

def create_tensors(df, window, input, output):
    df_to_np = df.to_numpy()
    # Sliding window on the data frame
    for i in range(len(df_to_np)-window):
        forward_pass = []

        for x in df_to_np[i:i + window]:
            forward_pass.append([x])

        input.append(forward_pass)
        output.append(df_to_np[i+window])

def create_batch(files, predictor, window):
    i = []
    o = []

    for file in files:
        path = os.path.join(directory_path, file)

        if os.path.isfile(path):
            df = pd.read_csv(path)
            create_tensors(df[predictor], window, i, o)

    return np.array(i), np.array(o)

window = 10
batches = 10
predictor = 'Close'

files = os.listdir(directory_path)
batch_size = round(len(files) / batches)

for i in range(batches):
    print((i * batch_size))
    print(batch_size * (i + 1))

    file_batch = files[(i * batch_size): batch_size * (i + 1)]

    input, output = create_batch(file_batch, predictor, window)

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

    model = None

    if not os.path.exists('model.keras'):
        model1 = Sequential()
        model1.add(InputLayer((window, 1)))
        model1.add(LSTM(128, dropout=0.2))
        model1.add(Dense(64, 'relu'))
        model1.add(Dropout(0.2))
        model1.add(Dense(1, 'linear'))
    else:
        model1 = load_model('model.keras')

    model1.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    cp1 = ModelCheckpoint('model.keras', save_best_only=True, verbose=1)
    model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
    model1.fit(input_train, output_train, validation_data=(input_validation, output_validation), epochs=5, callbacks=[early_stopping, cp1])

    model1 = load_model('model.keras')

    test_predictions = model1.predict(input_test).flatten()
    test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':output_test})

    print(test_results[len(test_results)-50:])
    test_results[len(test_results)-50:].plot()
    plt.show()