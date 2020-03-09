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
        self.resolutionX = resX
        self.resolutionY = resY
        self.featureSet = self.convertImagesToArray(self.trainingData, self.resolutionX, self.resolutionY)


    def createFeatureSetValidation(self):
        if len(self.featureSet) != 0:
            self.featureSetValidation = self.convertImagesToArray(self.testingData, self.resolutionX, self.resolutionY)
        else:
            print("[!] Feature Set not yet created")
        

    def createLabelSet(self):
        self.assignIndexToLabel()
        if len(self.labelSet) != 0:
            self.labelSet = []

        for label, image in self.trainingData:
            self.labelSet.append(self.labelDictionary[label])

        self.labelSet = np.array(self.labelSet)

    
    def createLabelSetValidation(self):
        if len(self.labelSet) != 0:
            if len(self.labelSetValidation) != 0:
                self.labelSetValidation = []

            for label, image in self.testingData:
                self.labelSetValidation.append(self.labelDictionary[label])

            self.labelSetValidation = np.array(self.labelSetValidation)
        else:
            print("[!] Label Set not yet created")


    def convertImagesToArray(self, setToConvert, resX, resY):
        if len(setToConvert) != 0:
            setToConvert = []

        for label, image in self.trainingData:
            imageResized = cv2.resize(image, (resX, resY))
            setToConvert.append(imageResized)

        setToConvert = np.array(setToConvert)
        if self.colorMode == True:
            setToConvert = setToConvert.reshape(len(setToConvert), resX, resY, 3)
        elif self.colorMode == False:
            setToConvert = setToConvert.reshape(len(setToConvert), resX, resY, 1)
        setToConvert = setToConvert / 255.0

        print("**Images converted, {} images, resized to {}x{}".format(len(setToConvert), resX, resY))
        return setToConvert


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


    def applyImageRotation(self, degree = None):
        rotation = (90, 180, 270)
        if degree != None:
            rotation = degree



    def applyImageScaling(self, scale = None):
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