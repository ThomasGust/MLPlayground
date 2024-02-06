import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import pickle

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential()

model.add(LSTM(170, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(170, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(170, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(170, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(160, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(160, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(160, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

save_model_input = input('Do you want to serialize this model?:')
save_model = None
model_name = None
model_number = None

if save_model_input in ('Yes', 'yes'):
    save_model = True
    model_name_input = input('What do you want to name this model?:')
    model_name = str(model_name_input)
    model_number_input = input('What model number do you want this model to be?:')
    model_number = str(model_number_input)
    print('Okay saving model')
    model.save(f'{model_name}{model_number}.model')

elif save_model_input in ('No', 'no'):
    print('Okay will not save model')
    save_model = False

else:
    print('Invalid response, will continue with saving dialogue')
    model_name_input = input('What do you want to name this model?:')
    model_name = str(model_name_input)
    model_number_input = input('What model number do you want this model to be?:')
    model_number = str(model_number_input)
    print('Okay saving model')
    model.save(f'{model_name}{model_number}.model')