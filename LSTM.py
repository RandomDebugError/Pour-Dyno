import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

patience = 100
learning_rate = 0.0005
num_epochs = 1000``

# read the data and generate sequences
raw_data = np.load('Robot_Trials_DL.npy')
#print(raw_data)
print(np.shape(raw_data))


# shuffle of the data
np.random.seed(0)
np.random.shuffle(raw_data)
print(np.shape(raw_data))


data_train = raw_data[0 : int(0.6 * np.shape(raw_data)[0])]
data_val = raw_data[int(0.6 * np.shape(raw_data)[0]) : int(0.8 * np.shape(raw_data)[0])]
data_test = raw_data[int(0.8 * np.shape(raw_data)[0]) : int(1.0 * np.shape(raw_data)[0])]

data_train_x = data_train[:, :,[0,2,3,4,5,6]]
data_train_y = data_train[:, :,[1]]
#print('Train_data_x is : ' + str(data_train_x))
#print('Train_data_y is : ' + str(data_train_y))
data_val_x = data_val[:, :,[0,2,3,4,5,6]]
data_val_y = data_val[:, :, 1]
data_test_x = data_test[:, :,[0,2,3,4,5,6]]
data_test_y = data_test[:, :, 1]


min = np.min(data_train)
max = np.max(data_train)

scaler = MinMaxScaler(feature_range=(min, max))
x = np.any(data_train, -1)

scaler.fit(data_train[x])
data_train[x] = scaler.transform(data_train[x])



# build the network structure
initializer = tf.keras.initializers.Zeros()
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, activation='tanh',return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(16, activation='tanh', return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(16, activation='tanh', return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation = 'tanh'))


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = 'MeanSquaredError', metrics = 'mse')


callbacks = [EarlyStopping(monitor='val_mse', patience=patience, verbose=1, mode='min'), 
            ModelCheckpoint(filepath='model.h5', verbose=1, monitor='val_mse', save_best_only=True, save_weights_only=False, mode='min')]


history = model.fit(data_train_x, data_train_y, batch_size = 50, epochs = num_epochs, validation_data = (data_val_x, data_val_y), callbacks=callbacks, shuffle = True)



result = model.predict(data_test_x)
plt.plot(result[0, :], 'r-', label = 'prediction')
plt.plot(data_test_y[0, :], 'b-', label = 'f(t)')
plt.legend()
plt.show()





