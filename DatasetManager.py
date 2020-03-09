import os
import pickle

import DataLoader
from DataSetModule import DataSetModule

class DatasetManager():
    def __init__(self):
        self.dataSets = []

    def printAvailableDatasets(self):
        print("All available datasets:")
        if (len(self.dataSets) != 0):
            for i in range(0, len(self.dataSets)):
                try:
                    print("{}. {}".format(i, self.dataSets[i].name), end = '')
                    if (len(self.dataSets[i].featureSet) == 0):
                        print(" WARNING: Feature set not yet created!")
                    else:
                        print("")
                except:
                    print("[!] There was an error trying to retrieve available datasets")
        else:
            print("[!] EMPTY")

    def createNewDataset(self):
        #Insert details about dataset
        print("Name: ", end = "")
        name = input()

        print("Path: ", end = "")
        path = input()

        print("Label file name: ", end = "")
        labelFile = input()

        print("Verification label file name: ", end = '')
        verificationFile = input()

        print("Color mode (y/n): ", end = "")
        colorMode = input()
        if colorMode == 'y':
            colorMode = True
        else:
            colorMode = False

        #Check if the label file exists before adding it to list
        try:
            open(os.path.join(path, labelFile))
        except:
            print("[!] Label file not existing")
            return

        #Create and load new dataset
        #TODO Refractor to use static methods!
        if verificationFile == "":
            loadedImagesTraining = DataLoader.loadDataset(path, labelFile, colorMode)

            newDSM = DataSetModule(name, colorMode, loadedImagesTraining, None)
            self.dataSets.append(newDSM)
        else:
            loadedImagesTraining = DataLoader.loadDataset(path, labelFile, colorMode)

            loadedImagesVerification = DataLoader.loadDataset(path, verificationFile, colorMode)
            
            newDSM = DataSetModule(name, colorMode, loadedImagesTraining, loadedImagesVerification)
            self.dataSets.append(newDSM)


    def printAvailableCommands(self):
        print("\n======DATASET MENU=======")
        print("0. Create training feature and label sets")
        print("1. Create validation feature and label sets")
        print("2. Save dataset to memory")
        print("3. Add validation feature set")
        print("4. Data Augumentation options")


    def manageDataset(self, datasetID):
        try:
            workingDataset = self.dataSets[datasetID]
        except:
            print("[!] Dataset not found or out of range")
            return

        while True:
            self.printAvailableCommands()
            keyIn = input()
            print("")
            if keyIn == '0':
                print("ResolutionX: ", end = '')
                resX = int(input())
                print("ResolutionY: ", end = '')
                resY = int(input())
                workingDataset.createFeatureSet(resX, resY)
                workingDataset.createLabelSet()
            elif keyIn == '1':
                print("[!] Not implemented yet")
            elif keyIn == '2':
                print("Filename: ", end = "")
                keyIn = input()
                self.dataSets[datasetID].saveDataSetModule(name = keyIn)
            elif keyIn == '3':
                print("Verification label file name: ", end = '')
                verificationFile = input()
                #Refractor to use static methods
            elif keyIn == '4':
                print("0. Image rotation")
                print("1. Image scaling")
                keyIn == input()
                if keyIn == '0':
                    self.rotateImages()
                elif keyIn == '1':
                    self.scaleImages()
                else:
                    print("[!] Invalid input")
            elif keyIn == 'd':
                self.dataSets[datasetID].printOutDebug()
            elif keyIn == 'e':
                break
            else:
                print("[!] Invalid option")
        

    def loadDatasetFromMemory(self):
            counter = 0
            savedDatasets = self.getSavedDatasets()
            for dsmFile in savedDatasets:
                print("{}. {}".format(counter, dsmFile))
                counter += 1
            if counter == 0:
                print("[!] No datasets found in memory")
                return
            print("a - Load All")
            print("Dataset to load: ", end = '')
            keyIn = input()
            print("")
            if (keyIn == 'a'):
                for dataset in savedDatasets:
                    try:
                        DSMload = pickle.load(open(dataset, 'rb'))
                        self.dataSets.append(DSMload)
                        print('**Dataset "{}" loaded'.format(DSMload.name))
                    except:
                        print("[!] DataSetModule file not found or could not be loaded!")
            else:
                try:
                    DSMload = pickle.load(open(savedDatasets[int(keyIn)], 'rb'))
                    self.dataSets.append(DSMload)
                    print('**Dataset "{}" loaded'.format(DSMload.name))
                except:
                    print("[!] DataSetModule file not found or could not be loaded!")


    def rotateImages(self):
        print("[!] Not implemented yet")


    def scaleImages(self):
        print("[!] Not implemented yet")


    def getSavedDatasets(self):
        datasets = []
        for file in os.listdir():
            if ".dsm" in file:
                datasets.append(file)
        return datasets


    def deleteDataset(self, id):
        try:
            del self.dataSets[id]
        except:
            print("[!] Could not delete dataset with id {}".format(id))

    def getDatasetByIndex(self, index):
        return self.dataSets[index]