import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv('iris.data', names=cols)

unique_classes = data['class'].unique()

for idx, name in enumerate(unique_classes):
    data.loc[ data['class'] == name, 'class' ] = idx

data['class'] = data['class'].astype(int)

# Stratified split: 80% train, 10% valid, 10% test, preserving class distribution
train, temp = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)
valid, test = train_test_split(temp, test_size=0.5, stratify=temp['class'], random_state=42)
# train, valid, test = np.split(data.sample(frac=1), [int(.8*len(data)), int(.9*len(data))])


def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    Y = np.reshape(y, (-1, 1))
    data = np.hstack(( X, Y ))
    return data, X, y

train, X_train, y_train = scale_dataset(train)
test, X_test, y_test = scale_dataset(test)
valid, X_valid, y_valid = scale_dataset(valid)


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

print(classification_report(y_test, y_pred))