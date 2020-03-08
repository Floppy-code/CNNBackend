import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from InputParser import InputParser
from DatasetManager import DatasetManager
from NeuralNetManager import NeuralNetManager

def main():
    DSManager = DatasetManager()
    NNManager = NeuralNetManager(DSManager)

    parser = InputParser(DSManager, NNManager)
    parser.runParser()

if __name__ == "__main__":
    main()