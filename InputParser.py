from DatasetManager import DatasetManager
from NeuralNetManager import NeuralNetManager

class InputParser():

    def __init__(self, DSManager, NNManager):
        self.DSManager = DSManager
        self.NNManager = NNManager
    

    def runParser(self):
        while True:
            self.printAvailableCommands()

            print("Input: ", end = "")
            keyboardIn = input()
            print("")

            if keyboardIn == '0':
                self.DSManager.printAvailableDatasets()

            elif keyboardIn == '1':
                self.DSManager.createNewDataset()

            elif keyboardIn == '2':
                self.DSManager.loadDatasetFromMemory()

            elif keyboardIn == '3':
                print("Dataset ID: ", end = "")
                keyboardIn = input()
                print("")
                self.DSManager.deleteDataset(int(keyboardIn))

            elif keyboardIn == '4':
                print("Dataset ID: ", end = "")
                keyboardIn = input()
                print("")
                self.DSManager.manageDataset(int(keyboardIn))

            elif keyboardIn == '5':
                self.NNManager.printAvailableNetworks()
            elif keyboardIn == '6':
                self.NNManager.createNewNetwork()
            elif keyboardIn == '7':
                self.NNManager.loadNetworkFromMemory()
            elif keyboardIn == '8':
                pass
            elif keyboardIn == '9':
                self.NNManager.manageNetwork()
            elif keyboardIn == 'e':
                break
            else:
                print("[!] Invalid input")

    def printAvailableCommands(self):
        print("\n========MAIN MENU=========")
        print("0. Show available datasets")
        print("1. Create new dataset")
        print("2. Load dataset from memory")
        print("3. Delete existing dataset")
        print("4. Manage existing dataset")
        print("5. List existing neural networks")
        print("6. Create new neural network")
        print("7. Load neural network")
        print("8. Delete neural network")
        print("9. Manage existing neural network")
        