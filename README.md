# Interactive LeNet
<p align="center">
  <img src="https://github.com/earnesdm/Interactive-LeNet-Demo/blob/main/img/LeNet_demo_4.png?raw=true"
width="400"
/>
</p>

## Description
An interactive program that lets you draw digits (0-9) with your mouse and send them to a trained LeNet model, which predicts the digit given the raw pixels. It is an enjoyable way to interact with LeNet and a great tool for teaching CNNs.  

The code is written in such a way that it is easy to try out new neural network architectures. The network architecture can be modified in "model.py" and retrained by running "train.py" which will save the new model parameters. Replace the old model parameters with the new ones and you can then rerun the GUI with "main.py."

To learn more about LeNet, read: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

Video Demo: https://youtube.com/shorts/avoPbmiHj-U?feature=share

## Getting Started
To use this project, run main.py. This will open the GUI in a new window. The main window of the GUI is a canvas that you can use to draw a digit (0-9) with your mouse. Once you have drawn your digit, pressing "Guess" will send the digit to LeNet (a convolutional neural network) who will guess your digit, given the drawing. Pressing "Clear" will clear the canvas, allowing you to draw a new digit.

Requires Python 3.10, PyQt6 6.8.0, torch 2.5.1, torchvision 0.20.1, numpy 2.2.1, and matplotlib 3.10.0
