import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

#TODO pridat aj 1D a 3D??
class DataSetModule():
    def __init__(self, name, colorMode, trainData, testData):
        self.name = name
        self.colorMode = colorMode  #True - RGB : False - BW
        self.resolutionX = 0
        self.resolutionY = 0
        self.trainingData = trainData #format -> [labels, cv2 image]
        self.validationData = testData  #format -> [labels, cv2 image]
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
        if len(self.validationData) == 0:
            print("[!] Cannot create feature set without input")
            return
        if len(self.featureSet) != 0:
            self.featureSetValidation = self.convertImagesToArray(self.validationData, self.resolutionX, self.resolutionY)
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
        if len(self.validationData) == 0:
            print("[!] Cannot create label set without input")
            return
        if len(self.labelSet) != 0:
            if len(self.labelSetValidation) != 0:
                self.labelSetValidation = []

            for label, image in self.validationData:
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


    #TODO Find better name...
    def augumentImagesRotation(self):
        rotation = (90, 180, 270)

        oldImageCount = len(self.trainingData)
        counter = 0

        fsetAddon = []
        lsetAddon = []

        for label, image in self.trainingData:
            for angle in rotation:
                rotatedImage = self.rotateImage(image,angle)
                #self.trainingData.append((label, rotatedImage))
                
                #I was running out of RAM trying to save them at full size...
                img = cv2.resize(rotatedImage, (self.resolutionX, self.resolutionY))
                fsetAddon.append(img)
                lsetAddon.append(self.labelDictionary[label])

            if counter == oldImageCount:
                break
            if (counter % 100) == 0:
                print("**Rotated {} out of {} images".format(counter, oldImageCount))
            counter += 1

        #DEBUG - Out of RAM
        self.trainingData = []

        fsetAddon = np.array(fsetAddon)
        if self.colorMode == True:
            fsetAddon = fsetAddon.reshape(len(fsetAddon), self.resolutionX, self.resolutionY, 3)
        else:
            fsetAddon = fsetAddon.reshape(len(fsetAddon), self.resolutionX, self.resolutionY, 1)
        fsetAddon = fsetAddon / 255.0
        lsetAddon = np.array(lsetAddon)

        print("**Concating arrays!")
        self.featureSet = np.concatenate((self.featureSet, fsetAddon), axis = 0)
        self.labelSet = np.concatenate((self.labelSet, lsetAddon), axis = 0)

        print("**Images rotated!")



    def rotateImage(self, image, angle):
        #I followed this StackOverflow reply to program this image rotation part
        #https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides/47248339
        #TODO - Make this faster if angle is 90*n

        height, width = image.shape[:2]
        center = (width/2.0, height/2.0)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        absCos = abs(M[0,0])
        absSin = abs(M[0,1])

        boundW = int(height * absSin + width * absCos)
        boundH = int(height * absCos + width * absSin)

        M[0,2] += boundW/2 - center[0]
        M[1,2] += boundH/2 - center[1]

        rotatedImage = cv2.warpAffine(image, M, (boundW, boundH))
        return rotatedImage


    def augumentImagesScaling(self, scale = None):
        pass #TODO
         

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


    def showImage(self, indexInFSet):
        plt.imshow(self.featureSet[indexInFSet].reshape(self.resolutionX, self.resolutionY, 3))
        plt.show()