import os
import cv2
import numpy as np

from DataSetModule import DataSetModule

class DataLoader():

    # colorMode:
    # False - Black and White
    # True - RGB
    def __init__(self, datasetName, path, infoFile, colorMode = True):
        self.path = path
        self.name = datasetName
        self.infoFile = infoFile
        self.colorMode = colorMode
        self.loadedData = []


    #Loads dataset and returns it as a DataSetModule object
    def loadDataset(self, pathToDataset, infoFile, mode = "text"):
        if mode == "text":
            datasetElements = []
            try:
                txt = open(os.path.join(pathToDataset, infoFile), 'r')

                for line in txt:
                    wordsInLine = line.rstrip()
                    wordsInLine = wordsInLine.split(";")
                    datasetElements.append([wordsInLine[0], wordsInLine[1]])

                print("**Loaded {} paths, loading into memory...".format(len(datasetElements)))

                for label, imagePath in datasetElements:
                    try:
                        if self.colorMode == True:
                            imgFile = cv2.imread(os.path.join(pathToDataset, imagePath), cv2.IMREAD_COLOR)
                        else:
                            imgFile = cv2.imread(os.path.join(pathToDataset, imagePath), cv2.IMREAD_GRAYSCALE)
                        self.loadedData.append([label, imgFile])
                    except:
                        print("[!] Loading/resizing of image {} failed".format(imagePath))

                print("**Loaded {} images into memory".format(len(self.loadedData)))

                DSModule = DataSetModule(self.name, self.colorMode, self.loadedData, None)
                return DSModule

            except:
                print("[!] Unable to load data from this file")

        elif mode == "binary":
            pass
        else:
            print("[!] Mode unknown")

        return False