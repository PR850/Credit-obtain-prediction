import pandas as pd
from tensorflow.keras.layers import Dense, Activation
import keras
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf


def oneDense():
    df = pd.read_csv('prep_data.csv', sep=',', header=None)

    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    for i in range(7):
        name = "Kredyt_20e_relu_sigm_adam_bin_crossen_Dense_" + str(2 ** i)
        tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

        model = keras.Sequential()

        model.add(Dense(2 ** i))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X, Y, validation_split=0.2, batch_size=20, epochs=20, callbacks=[tensorboard])
        model.save(name)


def twoDense():
    X_test = pd.read_csv('X_test.csv', sep=',', header=None)
    df = pd.read_csv('prep_data.csv', sep=',', header=None)

    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    for i in range(7):
        for k in range(7):
            name = "Kredyt_20e_relu_sigm_adam_bin_crossen_Dense_" + str(2 ** i) + "_Dense_" + str(2 ** k)
            tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

            model = keras.Sequential()

            model.add(Dense(2 ** i))
            model.add(Activation('relu'))

            model.add(Dense(2 ** k))
            model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(X, Y, validation_split=0.2, batch_size=20, epochs=20, callbacks=[tensorboard])
            model.save(name)