#Imports
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping

#My imports
import NeuralNetModule

#TODO -Pro mode
def LayerCreatorParser(module):
    model = module.model
    dataset = module.datasetModule

    while True:
        printAvailableLayers()
        print("Input: ", end = '')
        keyIn = input()
        if keyIn == 'e':
            break
        keyInSplit = keyIn.split(" ")

        noOfLayers = len(model.layers)

        #try:
        if int(keyInSplit[0]) == 0:
            if int(keyInSplit[1]) == 0: #Activation
                addActivationLayer(model)

            elif int(keyInSplit[1]) == 1: #Dense
                if noOfLayers == 0:
                    print("Neurons: ", end = '')
                    neurons = input()
                    addDenseLayer(model, int(neurons), input_shape = dataset.getInputShape())
                else:
                    print("Neurons: ", end = '')
                    neurons = input()
                    addDenseLayer(model, int(neurons))

            elif int(keyInSplit[1]) == 2: #Dropout
                print("Dropout rate: ", end = '')
                rate = input()
                addDropoutLayer(model, float(rate))

            elif int(keyInSplit[1]) == 3: #Flatten
                addFlattenLayer(model)

        elif int(keyInSplit[0]) == 1:
            if int(keyInSplit[1]) == 0: #Conv1D
                pass

            elif int(keyInSplit[1]) == 1: #Conv2D
                print("Number of filters: ", end = '')
                filters = input()

                print("Kernel size: ", end = '')
                kSize = input()
                kernelSize = (int(kSize), int(kSize))

                print("Strides (1,1): ", end = '')
                stride = input()
                if stride == '':
                    stride = (1,1)
                else:
                    stride = (int(stride), int(stride))

                print("Padding ('valid'): ", end = '')
                padding = input()
                if padding == "":
                    padding = 'valid'

                if noOfLayers == 0:
                    add2DConvolutionalLayer(model, int(filters), kernelSize, stride, padding, input_shape = dataset.getInputShape())
                else:
                    add2DConvolutionalLayer(model, int(filters), kernelSize, stride, padding)

        elif int(keyInSplit[0]) == 2:
            if int(keyInSplit[1]) == 0: #MaxPooling1D
                pass

            elif int(keyInSplit[1]) == 1: #MaxPooling2D
                print("Pool size (2,2): ", end = '')
                poolsize = input()
                if poolsize == '':
                    poolsize = (2,2)
                else:
                    poolsize = (int(poolsize), int(poolsize))

                print("Strides (pSize, pSize): ", end = '')
                strides = input()
                if strides == '':
                    strides = poolsize
                else:
                    strides = (int(strides), int(strides))

                print("Padding: ", end = '')
                padding = input()
                if padding == '':
                    padding = 'valid'

                addMaxPoolong2DLayer(model, poolsize, strides, padding)

            elif int(keyInSplit[1]) == 2: #AvgPooling1D
                pass

            elif int(keyInSplit[1]) == 3: #AvgPooling2D
                pass
        #except:
        #    print("[!] Wrong input format")


def printAvailableLayers():
    print("\n========Available Layers==========")
    print("0. Core Layers:				2. Pooling layers:")
    print("	0. Activation layer			0. MaxPooling1D layer")
    print("	1. Dense layer				1. MaxPooling2D layer")
    print("	2. Dropout layer			2. AvgPooling1D layer")
    print("	3. Flatten layer			3. AvgPooling2D layer")
    print("1. Convolutional layers:")
    print("	0.Conv1D layer")
    print("	1.Conv2D layer")


#CORE LAYERS=======================
def addDenseLayer(model, noOfNeurons, input_shape = None):
    if input_shape is None:
        model.add(Dense(noOfNeurons))
    else:
        model.add(Dense(noOfNeurons, input_shape = input_shape))
    addActivationLayer(model)

def addActivationLayer(model):
    print("Activation function: ", end = '')
    activation = input()
    model.add(Activation(activation))

def addDropoutLayer(model, rate, noise_shape = None, seed = None):
    model.add(Dropout(rate, noise_shape, seed))

def addFlattenLayer(model):
    model.add(Flatten())


#CONVOLUTIONAL LAYERS===============
def add1DConvolutionalLayer(model):
    pass

def add2DConvolutionalLayer(model, filters, kernel_size, strides = (1,1), padding = 'valid', input_shape = None): #Default values taken from keras documentation
    if input_shape is None:
        model.add(Conv2D(filters, kernel_size, strides = strides, padding = padding))
    else:
        model.add(Conv2D(filters, kernel_size, strides = strides, padding = padding, input_shape = input_shape))
    addActivationLayer(model)

    
#POOLING LAYERS=====================
def addMaxPooling1DLayer(model, pool_size = 2, strides = None, padding = 'valid'):
    pass

def addMaxPoolong2DLayer(model, pool_size, strides = None, padding = 'valid'):
    model.add(MaxPooling2D(pool_size, strides, padding))

def addAveragePooling1DLayer(model, pool_size, strides = None, padding = 'valid'):
    pass

def addAveragePooling1DLayer(model, pool_size, strides = None, padding = 'valid'):
    model.add(AveragePooling2D(pool_size, strides, padding))


#UPSAMPLING LAYERS===================