import tensorflow as tf
from tensorflow import keras
import numpy as np

classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Build the model
model = keras.Sequential()

# Add a few convolutional layers
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Add a few fully connected layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          batch_size=32,
          epochs=15,
          validation_data=(x_test, y_test))

# Evaluate the model on the test set
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Make predictions on new data
print("Example of input of image: airplane.jpg")
while(True):
    file = input("Enter image name located in the Images folder, type n to stop: ")
    if file == 'n':
        break
    path = "Images/" + file
    try:
        img = tf.keras.utils.load_img(path,target_size=(32,32))
        x_new = np.array(img)
        x_new = x_new[np.newaxis, :]
        predictions = model.predict(x_new)
        predicted_classes = np.argmax(predictions, axis=1)
        print("Class: ", classes[predicted_classes[0]])
    except:
        print("Invalid file name, try again.")
