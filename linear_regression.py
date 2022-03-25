import pandas as pd
import numpy as np
file = pd.read_csv('csv_file')
X_values = file.drop('y', axis=1)
y_value = file.y

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_values, y_value)

x= np.array([1,2,3,4,5]).reshape(-1,1)
lr.predict(x)


