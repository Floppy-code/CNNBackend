import os
import pickle
#import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping

import NNLayerCreator

lossFunctions = {0 : "categorical_crossentropy", 1 : "sparse_categorical_crossentropy", 2 : "binary_crossentropy"}
optimizers = {0 : "Adam", 1 : "Adamax", 2 : "Nadam", 3 : "SGD"}

class NeuralNetModule():
    def __init__(self, name, dataset):
        self.name = name
        self.model = Sequential()
        self.datasetModule = dataset


    def buildModel(self):
        NNLayerCreator.LayerCreatorParser(self)


    def compileModel(self):
        print("Optimizers: ")
        for i in range(0, len(optimizers)):
            print("{}. {}".format(i, optimizers[i]))
        print("Optimizer: ", end = '')
        optimizerIndex = input()
        print("")

        print("Loss functions: ")
        for i in range(0, len(lossFunctions)):
            print("{}. {}".format(i, lossFunctions[i]))
        print("Loss function: ", end = '')
        lossFunctionIndex = input()
        print("")

        self.model.compile(optimizers[int(optimizerIndex)], loss = lossFunctions[int(lossFunctionIndex)], metrics = ['accuracy'])
        self.model.summary()


    def fitModel(self):
        print("Epochs (10): ", end = '')
        epochs = input()
        if epochs == '':
            epochs = 10

        print("Batch size (64): ", end = '')
        batchSize = input()
        if batchSize == '':
            batchSize = 64

        print("Validation split (0.1): ", end = '')
        valSplit = input()
        if valSplit == '':
            valSplit = 0.1

        self.model.fit(self.datasetModule.featureSet, self.datasetModule.labelSet, epochs = int(epochs), batch_size = int(batchSize), validation_split = float(valSplit))


    def predict(self, setToPredict):
        results = self.model.predict(setToPredict, verbose = 1)
        return results

    def saveModel(self, name):
        name = name + ".nnm"
        pickleOut = open(name, 'wb')
        pickle.dump(self, pickleOut)
        print("**Neural Net Module saved as {}".format(name))