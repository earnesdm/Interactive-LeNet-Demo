# Interactive LeNet
<p align="center">
  <img src="https://github.com/earnesdm/Interactive-LeNet-Demo/blob/main/img/LeNet_demo_4.png?raw=true"
width="400"
/>
</p>

## Description
A interactive program that lets you draw digits (0-9) with you mouse and send them to a trained LeNet model which predicts the digit given the raw pixels. It is an enjoyable way to interact with LeNet and a great tool for teaching CNNs.  

The code is written in such a way that it is easy to try out new neural network architechtures. The network architechture can be modified in "model.py" and retrained by running "train.py" which will overwrite the old model parameters. You can then rerun the GUI with "main.py."

To learn more about LeNet read: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

## Getting Started
To use this project run main.py. This will open the GUI in a new window. The main window of the GUI is a canvas that you can use to draw a digit (0-9) with your mouse. Once you have drawn your digit, pressing "Guess" will send the digit to LeNet (a convolutional neural network) who will guess your digit, given the drawing. Pressing "Clear" will clear the canvas, allowing you to draw a new digit.
