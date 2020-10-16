# MultiLayeredPerceptron
Implementing Multi Layered Perceptron from scratch in python


## MNIST Dataset
The MNIST database of handwritten digits, it has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

[Source](http://yann.lecun.com/exdb/mnist/)

Sample Image from the dataset
<a href="https://imgur.com/0jU7Odt"><img src="https://i.imgur.com/0jU7Odt.png" title="source: imgur.com" /></a>

## Network Architecture
```
# Model Architecture
inputs = 784
layers = [32, 16]
output = 10
mlp = Multi_Layered_Perceptron(n_inputs=inputs, hidden_layers=layers, n_outputs=output, activation_function='sigmoid')
```
<a href="https://imgur.com/K9iCVfl"><img src="https://i.imgur.com/K9iCVfl.png" title="source: imgur.com" /></a>

## Training
```
mlp.GradientDescent(X_train, Y_train, learning_rate=0.01, epochs=100,  beta1=0.9, beta2=0.999)
mlp.plot_loss()
```
<a href="https://imgur.com/WZUeS74"><img src="https://i.imgur.com/WZUeS74.png" title="source: imgur.com" /></a>



## Evaluation
```
y_pred_train_classes = mlp.predict(X_train.values)
y_pred_test_classes = mlp.predict(X_test.values)
```
<a href="https://imgur.com/KSDfcUE"><img src="https://i.imgur.com/KSDfcUE.png" title="source: imgur.com" /></a>

