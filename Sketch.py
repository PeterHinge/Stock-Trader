import pandas as pd
import numpy as np
import random
import time
from sklearn import preprocessing
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint


DATA_NAME = 'AAPL'
DATA_VALUES = ['Date', 'Open', 'High', 'Low', 'close', 'Adj Close', 'volume']
VALUES_TO_LEARN_FROM = ['close', 'volume']
DATA_REVERSED = False

SEQ_LEN = 50
FUTURE_PERIOD_PREDICT = 10
EPOCHS = 15
BATCH_SIZE = 64
VALIDATION_SIZE = 0.1
MODEL_NAME = f"{DATA_NAME}-{int(time.time())}"


def targeting(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocessing_df(df):
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    print(df.head(30))

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


pd.set_option('display.max_columns', None)

data = pd.read_csv(f'data/{DATA_NAME}.csv', names=DATA_VALUES)

df = data[VALUES_TO_LEARN_FROM]
if DATA_REVERSED:
    df = df[::-1]
    df = df.reset_index(drop=True)

df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)

df['future'] = df['close'].shift(-FUTURE_PERIOD_PREDICT)
df['target'] = list(map(targeting, df['close'], df['future']))
df = df.drop("future", 1)

df.dropna(inplace=True)

train_df = df[(df.index <= int(len(df.index) * (1 - VALIDATION_SIZE)))]
validation_df = df[(df.index > int(len(df.index) * (1 - VALIDATION_SIZE)))]

print(train_df.head(30))
print(validation_df.head())


train_x, train_y = preprocessing_df(train_df)
validation_x, validation_y = preprocessing_df(validation_df)

print(f"Training data: {len(train_x)} Validation data: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, Buys: {train_y.count(1)}")
print(f"VALIDATION - Dont buys: {validation_y.count(0)}, Buys: {validation_y.count(1)}")


model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(MODEL_NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))


history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("models/{}".format(MODEL_NAME))
