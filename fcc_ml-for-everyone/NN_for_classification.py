import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib


model_name = "iris_model.h5"
scaler_name = "scaler.pkl"
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

data = pd.read_csv('iris.data', names=cols)
unique_classes = data['class'].unique()


for idx, name in enumerate(unique_classes):
    data.loc[ data['class'] == name, 'class' ] = idx

data['class'] = data['class'].astype(int)


# Stratified split: 80% train, 10% valid, 10% test, preserving class distribution
train, valid = train_test_split(data, test_size=0.1, stratify=data['class'], random_state=42)
# valid, test = train_test_split(temp, test_size=0.5, stratify=temp['class'], random_state=42)



def scale_dataset(dataframe, scaler=None):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(x)
    else:
        X = scaler.transform(x)

    return X, y, scaler



def plot_history(title, history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  plt.title(title, color='green')

  ax1.plot(history.history['loss'], label='loss', color='blue')
  ax1.plot(history.history['val_loss'], label='val_loss', color='red')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('sparse_categorical_crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy', color='blue')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy', color='red')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  plt.show()


def train_model(num_nodes, learning_rate, epochs, batch_size):
    X_train, y_train, scaler = scale_dataset(train)
    X_valid, y_valid, _ = scale_dataset(valid, scaler)

    joblib.dump(scaler, "scaler.pkl")
    
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(units=num_nodes, activation='relu'),
        tf.keras.layers.Dense(units=num_nodes, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), verbose=0)
    nn_model.save(model_name)

    return history


# Test for Different nodes, epochs, batch, learning_rate
def brute_train():
    for num_nodes in [10, 20, 40, 60]:
        for learning_rate in [0.01, 0.005, 0.001]:
            for epochs in [50, 100, 200]:
                for batch_size in [8, 16, 32]:
                    title = f'node {num_nodes}, LR {learning_rate}, Ep {epochs}, Bch {batch_size}'
                    nn_model, history = train_model(num_nodes, learning_rate, epochs, batch_size)
                    plot_history(title, history)



def print_weight_bias():
    model = tf.keras.models.load_model(model_name)
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        
        if len(weights) == 0:
            continue  # skip layers without weights (like Input)

        w, b = weights  # weights and bias
        
        print(f"\nLayer {i}: {layer.name}")
        print(f"Weights shape: {w.shape}")
        print(f"Bias shape: {b.shape}")
        
        print("Weights:\n", w)
        print("Bias:\n", b)

# print_weight_bias()


def predict_flower():
    model = tf.keras.models.load_model(model_name)
    scaler = joblib.load(scaler_name)

    class_names = unique_classes

    # Take input
    sepal_length = float(input("Sepal length: "))
    sepal_width = float(input("Sepal width: "))
    petal_length = float(input("Petal length: "))
    petal_width = float(input("Petal width: "))

    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    pred_probs = model.predict(input_scaled)
    pred_class = np.argmax(pred_probs, axis=1)[0]

    print("Predicted class:", class_names[pred_class])



# Train Model Here and save it
# history = train_model(10, 0.001, 100, 10)
# plot_history("", history)


predict_flower()



# pred_probs = nn_model.predict(X_valid)
# pred_classes = np.argmax(pred_probs, axis=1)
# print(pred_probs[:10])
# print(pred_classes[:10])
# print(classification_report(y_valid, pred_classes))