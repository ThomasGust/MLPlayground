import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow.keras.datasets as tf_datasets
from tensorflow.keras.losses import SparseCategoricalCrossentropy

#all you need for this to run is tensorflow which is
#installed by using:
#pip install tensorflow
#(as of now tensorflow only runs with a 64 bit python environment)

#This is a implementation of a recurrent neural network (rnn)
#that uses lstm (long short term memory) cells on the mnist dataset

#its architecture is quite simple.
#As you can see below the first part of it is composed
#of lstm cells, and then the second part
#is just made up of dense layers.

#if you want to use this on your own data go right ahead. This script does not
#include preprocessing though so you will need to do that yourself.

#On my test with the mnist dataset I could get into the high 90s
#after just 4 epochs

#if you do change the dataset you should probably mess around
#with the hyperparameters of the model to see what works best
#for your data.
#if your images are bigger feel free to add more layers to the
#model. You should also try messing around with the size of the
#layers.

#if you want to save the model for later use,
#use: insert_model_name_here.save(parameters)
#insert_model_name_here.save(parameters)
#to save the full model, or just the weights
#respectively


NUM_EPOCHS = 4
LEARNING_RATE = 1e-3
NUM_CLASSES = 10 #you need to change this depending on the number of classes
#in your dataset.




mnist = tf_datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0
x_train = x_train / 255.0

model = Sequential()

model.add(LSTM(85, activation='relu', return_sequences=True,  input_shape=(x_train.shape[1:])))
model.add(Dropout(0.2))

model.add(LSTM(85, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(85, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(85, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(160, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(160, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(160, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(NUM_CLASSES, activation='softmax'))

network_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=network_optimizer,
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=NUM_EPOCHS, validation_data=(x_test, y_test))

#model.save_weights('insert_model_name_here')