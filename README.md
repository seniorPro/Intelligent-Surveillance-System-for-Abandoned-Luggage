# Intelligent Surveillance System for Abandoned Luggage 

This repository includes the codes and documents for the application called Intelligent Surveillance System for Abandoned Luggage.
Details are given below.

__Libraries:__
- Numpy
- multiprocessing 
- OpenCV 3.5
- Tensorflow 1.3

__Before working on the project:__

There are some requirements that need to be done to run this project. First of all, inside the installed folder of 
Tensorflow, models/research folder should be found or you can just download from the Tensorflow Object Detection Api repository 
via the link:
[Tensorflow Repository](https://github.com/tensorflow/models/tree/master/research/object_detection)

Then, if you have a trained model with a frozen model (with the extension .pb) of it, 
put this model file into 
models/research/object_detection/ folder with the text which is for the label names in the format of this API.(you can 
find the details in the given link).
