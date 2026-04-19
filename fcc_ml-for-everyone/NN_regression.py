# stop TensorFlow from trying GPU detection.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from matplotlib import axes
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression

dataset_cols = ["bike_count", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
# load csv and drop some columns from the column axis
df = pd.read_csv("SeoulBikeData.csv").drop(["Date", "Seasons", "Holiday", "Hour"], axis=1)

df.columns = dataset_cols
df['functional'] = (df['functional'] == "Yes").astype(int)
df = df.drop(["wind", "visibility", "functional"], axis=1)


# for label in df.columns[1:]:
# 	plt.scatter(df[label], df['bike_count'])
# 	plt.title(label)
# 	plt.ylabel("Bike count")
# 	plt.xlabel(label)
# 	plt.show()

# train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
df = df.sample(frac=1).reset_index(drop=True)
train = df[:int(0.8*len(df))]
test  = df[int(0.8*len(df)):]

def get_xy(df, x_cols, y_col):
    X = df[x_cols].values
    y = df[[y_col]].values
    return X, y


# # Single Linear Reg.
X_train_temp, y_train_temp = get_xy(train, ["temp"], "bike_count")
X_test_temp, y_test_temp = get_xy(test, ["temp"], "bike_count")

# temp_reg = LinearRegression()
# temp_reg.fit(X_train_temp, y_train_temp)

# # print(temp_reg.coef_, temp_reg.intercept_, temp_reg.score(X_test_temp, y_test_temp))
# plt.scatter(X_valid_temp, y_valid_temp, label="Data", color="blue")
# x = tf.linspace(-20, 40, 100).numpy().reshape(-1, 1)
# plt.plot(X_test_temp, temp_reg.predict(X_test_temp), label="Fit", color="red", linewidth=3)
# plt.legend()
# plt.title("bike vs temp")
# plt.ylabel("Number of bikes")
# plt.xlabel("Temp")
# plt.show()

# Multiple Linear Reg.
X_train_all, y_train_all = get_xy(train, df.columns[1:], "bike_count")
X_test_all, y_test_all = get_xy(test, df.columns[1:], "bike_count")

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)
print(all_reg.score(X_test_all, y_test_all))