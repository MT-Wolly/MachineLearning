# -*- coding: utf-8 -*-
#Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

"""
This deep learning model is used to predict the net power output for a combined cycle power plant,
given the ambient temperature, pressure, relative humidity and exhaust vacuum.
A combined cycle power plant (CCPP) is composed of gas turbines (GT), steam turbines (ST) and 
heat recovery steam generators. In a Combined cycle power Plant, the electricity is generated by gas and steam turbines, 
which are combined in one cycle, and is transferred from one turbine to another. 
While the Vacuum is colected from and has effect on the Steam Turbine, the other three of the ambient
variables affect the performance.
Temperature (T) in the range 1.81°C and 37.11°C,
- Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
- Relative Humidity (RH) in the range 25.56% to 100.16%
- Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
- Net hourly electrical energy output (EP) 420.26-495.76 MW
##The averages are taken from various sensors located around the plant that record the ambient 
# variables every second. The variables are given without normalization
"""
data = pd.read_csv("Combine_Cycle.csv")
X = data.iloc[:, :-1 ].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units= 6, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer='adam', loss='mean_squared_error')

ann.fit(X_train, y_train, batch_size=32, epochs = 100)

y_pred = ann.predict(X_test)
np.set_printoptions(precision =2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))