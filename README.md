# Simple Image Classification CNN using Tensorflow

## Requirements
### **Python 3**
If you need to download Python 3 use this link to download: https://www.python.org/downloads/
### **Tensorflow**
To install Tensorflow run the following command in a terminal:
```
pip3 install tensorflow
```
### **Numpy**
To install Numpy run the follwing command in a terminal:
```
pip3 install numpy
```

## How to run it
Clone the repo and inside the folder on your machine, run the python script main.py in a terminal
```
python3 main.py
```
When prompted enter a file which you want to classify.
```
Example of input of image: airplane.jpg
Enter image name located in the Images folder, type n to stop: airplane1.jpg
```

## How it works
This is a simple image classification convolutional neural network made by using tensorflow. The network is trained on the CIFAR10 dataset which is a dataset of 32x32 colored images for 10 different classes including: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This dataset includes 60000 training images and 10000 test images. The images are then normalized by dividing each pixel by 255. The labels are converted to one-hot encoding. The model use ReLU activation for the convolutional layers and softmax for the output layer. The model is then complied with categorical cross-entropy loss function, Adam optimization algorithm, and the accuracy metric. Currently the model is trained for 15 epochs but you can change that as you see fit.
