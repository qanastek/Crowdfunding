from typing import List

class Dataset:

    def __init__(self, path):

        # Train set
        self.x_train = None
        self.y_train = None

        # Test set
        self.x_test = None
        self.y_test = None

        # Labels
        self.labels = None

        # Load the corpora
        self.__load(path)
    
    def __load(self, path, cache=False):
        print("Please implement the loading function for the dataset!")
        raise NotImplementedError
