import os
import cv2
import numpy as np

# colorMode:
# False - Black and White
# True - RGB

#Loads images from dataset and returns them in array
def loadDataset(pathToDataset, infoFile, colorMode = True, mode = "text"):
    loadedData = []
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
                    if colorMode == True:
                        imgFile = cv2.imread(os.path.join(pathToDataset, imagePath), cv2.IMREAD_COLOR)
                    else:
                        imgFile = cv2.imread(os.path.join(pathToDataset, imagePath), cv2.IMREAD_GRAYSCALE)
                    loadedData.append((label, imgFile))
                except:
                    print("[!] Loading/resizing of image {} failed".format(imagePath))

            print("**Loaded {} images into memory".format(len(loadedData)))

            return loadedData

        except:
            print("[!] Unable to load data from this file")

    elif mode == "binary":
        pass
    else:
        raise Exception("Invalid mode used!")

    return False