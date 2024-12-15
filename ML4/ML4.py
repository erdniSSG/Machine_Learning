import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img

#load adn split data
X = np.load('Xtrain_classification2.npy')
y = np.load('ytrain_classification2.npy')
Xtest = np.load('Xtest_classification2.npy')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#reshape images back to 3D tensor
X_train_reshaped = X_train.reshape(-1, 28, 28, 3)
X_val_reshaped = X_val.reshape(-1, 28, 28, 3)

#resizing
X_train = np.array([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in X_train_reshaped])
X_val = np.array([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in X_val_reshaped])
Xtest = np.array([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in Xtest.reshape(-1, 28, 28, 3)])

#convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=6)
y_val = to_categorical(y_val, num_classes=6)

#normalizing images
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
Xtest = Xtest.astype('float32') / 255.0

#data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#calculate class weights
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))


#feature extraction using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))  
for layer in base_model.layers:
    layer.trainable = False

#model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(X_train, y_train), epochs=20, validation_data=(X_val, y_val), class_weight=class_weights)

#predictions
y_pred = model.predict(X_val)
y_test_pred = model.predict(Xtest)

y_pred_class = np.argmax(y_pred, axis=1)
y_test_pred_class = np.argmax(y_test_pred, axis=1)

#compute metrics
print("Balanced Accuracy:", balanced_accuracy_score(np.argmax(y_val, axis=1), y_pred_class))
print(classification_report(np.argmax(y_val, axis=1), y_pred_class))

#plot th econfusion matrix
cm = confusion_matrix(np.argmax(y_val, axis=1), y_pred_class)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

np.save('test_predictions.npy', y_test_pred_class)
