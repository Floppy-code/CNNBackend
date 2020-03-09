import cv2
import os
import numpy as np
import pickle

from NeuralNetModule import NeuralNetModule
import NNLayerCreator

class NeuralNetManager():
    def __init__(self, DSManager):
        self.DSManager = DSManager
        self.neuralNetworks = []

    def printAvailableNetworks(self):
        print("All available neural network modules:")
        if len(self.neuralNetworks) == 0:
            print("[!] EMPTY")
        else:
            for i in range(0, len(self.neuralNetworks)):
                print("{}. {}".format(i, self.neuralNetworks[i].name))

    def createNewNetwork(self):
        print("Name: ", end = '')
        name = input()
        self.DSManager.printAvailableDatasets()
        print("Dataset to use [index]: ", end = '')
        dataset = input()
        try:
            newNetwork = NeuralNetModule(name, self.DSManager.getDatasetByIndex(int(dataset)))
            self.neuralNetworks.append(newNetwork)
            print("** Neural network module \"{}\" created!".format(name))
        except:
            print("[!] Failed to create a new neural network module")


    def manageNetwork(self):
        self.printAvailableNetworks()
        print("Network module: ", end = '')
        moduleIndex = input()
        module = None
        try:
            module = self.neuralNetworks[int(moduleIndex)]
        except:
            print("[!] Module not found or index out of range")
            return
        
        while True:
            self.printAvailableCommands()
            print("Input: ", end = '')
            keyIn = input()
            if keyIn == '0':
                module.buildModel()
            elif keyIn == '1':
                module.compileModel()
            elif keyIn == '2':
                module.fitModel()
            elif keyIn == '3':
                self.predictSingle(module)
            elif keyIn == '4':
                pass
            elif keyIn == '5':
                print("Save as deep copy? (y/n): ", end = '')
                keyIn = input()
                if "y" in keyIn:
                    print("Filename: ", end = '')
                    keyIn = input()
                    module.saveModelDeep(keyIn)
                elif "n" in keyIn:
                    print("Filename: ", end = '')
                    keyIn = input()
                    module.saveModelShallow(keyIn)
                else:
                    print("[!] Invalid input")
            elif keyIn == 'e':
                break
            else:
                print("[!] Invalid option")

    def printAvailableCommands(self):
        print("\n========NETWORK MODULE MENU=========")
        print("0. Build model")
        print("1. Compile model")
        print("2. Fit model")
        print("3. Predict (single input)")
        print("4. Predict dataset")
        print("5. Save network module to memory")
        print("e. Back")


    def predictSingle(self, module):
        desiredInputShape = module.datasetModule.getInputShape()
        resX = desiredInputShape[0]
        resY = desiredInputShape[1]

        print("Path: ", end = '')
        path = input()
        imgFile = None
        try:
            if (module.datasetModule.colorMode == True):
                imgFile = cv2.imread(path, cv2.IMREAD_COLOR)
                imgFile = cv2.resize(imgFile, (resX, resY))
            else:
                imgFile = cv2.imread(path, cv2.IMREAD_COLOR)
                imgFile = cv2.resize(imgFile, (resX, resY))
        except:
            print("[!] Failed to load/resize file {}".format(path))
            return

        array = [imgFile]
        npArray = np.array(array)
        if (module.datasetModule.colorMode == True):
            npArray = npArray.reshape(1, resX, resY, 3)
        else:
            npArray = npArray.reshape(1, resX, resY, 1)

        result = module.predict(npArray)
        print("Probabilities: ")
        for i in range(0, len(result[0])):
            print("{}% - {}".format(result[0][i] * 100, module.datasetModule.getNameFromLabel(i)))


    def deleteExistingNetwork(self):
        pass


    def getSavedNeuralModules(self):
        savedModules = []
        for file in os.listdir():
            if ".nnm" in file or ".h5" in file:
                savedModules.append(file)
        return savedModules


    def loadNetworkFromMemory(self):
        counter = 0
        savedModules = self.getSavedNeuralModules()
        for nnmFile in savedModules:
            print("{}. {}".format(counter, nnmFile))
            counter += 1
        if counter == 0:
            print("[!] No neural modules found in memory")
            return
        print("a - Load All")
        print("Dataset to load: ", end = '')
        keyIn = input()
        print("")
        if (keyIn == 'a'):
            for module in savedModules:
                if ".nnm" in module:
                    try:
                        NNMLoad = pickle.load(open(module), 'rb')
                        self.neuralNetworks.append(NNMLoad)
                        print('**Module "{}" loaded'.format(NNMLoad.name))
                    except:
                        print("[!] NeuralNetModule file {} not found or could not be loaded!".format(module))
        else:
            if ".h5" in savedModules[int(keyIn)]:
                if len(self.neuralNetworks) == 0:
                    print("[!] No existing neural network modules, cannot assign .h5 file")
                    return
                toLoad = savedModules[int(keyIn)]
                self.printAvailableNetworks()
                print("Load .h5 into network: ", end = '')
                keyIn = input()
                try:
                    self.neuralNetworks[int(keyIn)].loadModelShallow(toLoad)
                except:
                    print("[!] .h5 model could not be loaded into neural network module \"{}\"".format(self.neuralNetworks[int(keyIn)].name))
            else:
                try:
                    NNMLoad = pickle.load(open(savedModules[int(keyIn)], 'rb'))
                    self.neuralNetworks.append(NNMLoad)
                    print('**Module "{}" loaded'.format(NNMLoad.name))
                except:
                    print("[!] NeuralNetModule file not found or could not be loaded!")