from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pandas as pd
import numpy as np

#if you want to run this copy, and paste this file into a .py file
#you will also need to pip install the following libraries:
#tensorflow: pip install tensorflow
#pandas: pip install pandas
#numpy: pip install numpy

#after that you should be all set

#if you try this out yourself I highly reccomend tinkering with the settings,
#and code. The ones that I put in are really just placeholders.

#I also can not have in depth comments for everything so please explore.

#The code will also not work without changing a couple of the values which are specific to the structure of your csv.
#A good example is when I put training_dataframe/testing_dataframe.csv_column_name.values or anything like that.
#make sure to change whatever csv_column_name that I put to whatever the name of the columns in your csv files.

#I can't explain everything but if you get stuck try to figure it out alone,
# and if you can't just do a google search or something.

TRAINING_DIR = '' #put relative path of training data (must be csv)
TESTING_DIR = '' #put relative path of testing data (must be csv)
INPUT_SHAPE = (2,)
NUM_CLASSES = 2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25

training_dataframe = pd.read_csv(TRAINING_DIR)
np.random.shuffle(training_dataframe.values)

model = keras.Sequential()

model.add(Dense(500, input_shape=INPUT_SHAPE, activation='relu'))

model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))

model.add(Dense(NUM_CLASSES, activation='softmax'))

network_optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
              optimizer=network_optimizer,
              metrics=['accuracy'])


x = np.column_stack((training_dataframe.x.values, training_dataframe.y.values))
model.fit([x], training_dataframe.color.values, epochs=NUM_EPOCHS)

testing_dataframe = pd.read_csv(TESTING_DIR)
test_x = np.column_stack((testing_dataframe.x.values, testing_dataframe.y.values))

print()
print()
print()
print('Evaluation:')
print(model.evaluate(test_x, testing_dataframe.color.values))

print('That first number was the loss of the model, the second one was its accuracy.')
