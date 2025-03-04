import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

#load data  
X = np.load("Xtrain_Classification1.npy") 
y = np.load("ytrain_Classification1.npy") 

print(X)

#split the data into training and validation sets
split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

#normalize images
X_train = tf.cast(X_train, tf.float32) / 255.0
X_val = tf.cast(X_val, tf.float32) / 255.0

#data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

def augment(image, label):
    image = data_augmentation(image)
    return image, label


#convert to TensorFlow datasets
dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset_train = dataset_train.map(augment).shuffle(1000).batch(32)
dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

#compute class weights
class_counts = tf.math.bincount(tf.cast(y_train, tf.int32)).numpy()  
total_samples = np.sum(class_counts)  
class_weights = {i: total_samples / (len(class_counts) * class_counts[i]) for i in range(len(class_counts)) if class_counts[i] > 0}


#defining model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model training 
model.fit(dataset_train, epochs=20, validation_data=dataset_val, class_weight=class_weights)

#predictions
y_pred = model.predict(X_val)
y_pred_class = tf.cast(y_pred > 0.5, tf.int32).numpy().flatten()

#evaluation metrics
balanced_acc = tf.keras.metrics.BinaryAccuracy()(y_val, y_pred_class).numpy()
print("Balanced Accuracy:", balanced_acc)

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred_class))

print(np.shape(np.load("Xtest_Classification1.npy")))
