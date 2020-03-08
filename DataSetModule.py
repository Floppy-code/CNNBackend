import pickle
import os
import numpy as np
import cv2

#TODO pridat aj 1D a 3D??
class DataSetModule():
    def __init__(self, name, colorMode, trainData, testData):
        self.name = name
        self.colorMode = colorMode  #True - RGB : False - BW
        self.resolutionX = 0
        self.resolutionY = 0
        self.trainingData = trainData #format -> [labels, cv2 image]
        self.testingData = testData  #format -> [labels, cv2 image]
        self.labelDictionary = {}
        self.reverseLabelDictionary = {}
        self.featureSet = []
        self.labelSet = []
        self.featureSetValidation = []
        self.labelSetValidation = []


    def createFeatureSet(self, resX, resY):
        if len(self.featureSet) != 0:
            self.featureSet = []

        for label, image in self.trainingData:
            imageResized = cv2.resize(image, (resX, resY))
            self.featureSet.append(imageResized)

        self.featureSet = np.array(self.featureSet)
        if self.colorMode == True:
            self.featureSet = self.featureSet.reshape(len(self.featureSet), resX, resY, 3)
        elif self.colorMode == False:
            self.featureSet = self.featureSet.reshape(len(self.featureSet), resX, resY, 1)
        self.featureSet = self.featureSet / 255.0

        self.resolutionX = resX
        self.resolutionY = resY

        print("**FeatureSet created, {} images, resized to {}x{}".format(len(self.featureSet), resX, resY))


    def createLabelSet(self):
        self.assignIndexToLabel()
        if len(self.labelSet) != 0:
            self.labelSet = []

        for label, image in self.trainingData:
            self.labelSet.append(self.labelDictionary[label])

        self.labelSet = np.array(self.labelSet)


    def assignIndexToLabel(self):
        counter = 0
        for label in self.getUniqueLabels():
            self.labelDictionary.update( {label : counter} )
            self.reverseLabelDictionary.update( {counter : label} )
            counter += 1
        print("Dictionary")
        print(self.labelDictionary)


    def getUniqueLabels(self):
        uniqueLabels = []
        for label, image in self.trainingData:
            if label not in uniqueLabels:
                uniqueLabels.append(label)
        print("Labels:")
        print(uniqueLabels)
        return uniqueLabels


    def createFeatureSetValidation(self):
        pass


    def createLabelSetValidation(self):
        pass
            

    def saveDataSetModule(self, path = "", name = "DSM"):
        name = name + ".dsm"
        saveAt = os.path.join(path, name)
        pickleOut = open(saveAt, 'wb')
        pickle.dump(self, pickleOut)
        print("**Data set module was saved at {}\n".format(saveAt))


    def getInputShape(self):
        if self.colorMode:
            return (self.resolutionX, self.resolutionY, 3)
        return (self.resolutionX, self.resolutionY, 1)


    def getNameFromLabel(self, label):
        return self.reverseLabelDictionary[label]


    #DEBUG
    def printOutDebug(self):
        print("Unique labels: {}".format(len(self.labelDictionary)))
        print("Items: {} Resolution: {}x{} Color mode: {}".format(len(self.featureSet), self.resolutionX, self.resolutionY, self.colorMode))
        print("Input shape for CNN: {}".format(self.getInputShape()))

    #TODO: Pridat metody na upravu datasetu (scaling, otacanie atd...)