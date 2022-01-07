import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf

test_data = np.load('Robot_Trials_DL_Test_30pct.npy')

test_x = test_data[:, :,[0,2,3,4,5,6]]
test_y = test_data[:, :, 1]

load_model = tf.keras.models.load_model('model.h5')
pred = load_model.predict(test_x,batch_size=1)



plt.plot(pred[0, :], 'r-', label = 'prediction')
plt.plot(test_y[0, :], 'b-', label = 'f(t)')
plt.legend()
plt.show()