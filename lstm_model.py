from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def build_model_and_train(x_train, y_train, x_val, y_val, max_len, d):
    print('start train')
    batch_size = 10
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, max_len, d[1])))
    model.add(LSTM(64, return_sequences=True, stateful=True))
    model.add(LSTM(64, stateful=True))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=50, shuffle=False, validation_data=(x_val, y_val))
    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")
    print("Saved model to disk")
