from abc import ABC, abstractmethod
import importlib

class Analyzer(ABC):
    @abstractmethod
    def __init__(self, jsonConfig):
        pass

    @abstractmethod
    def main(self, classTrackers):
        pass

    @abstractmethod
    def saveToSQL(self):
        pass



def loadAnalyzer(path):
    try:
        with open(path, 'r') as handle:
            analysisConfig = json.load(handle)
            objectType = analysisConfig["objectType"]
            objectConfig = analysisConfig["objectConfig"]
    except:
        print("Couldn't find the file")

    module = importlib.import_module(objectType)
    class_ = getattr(module, objectType.title())
    instance = class_(objectConfig)

    return instance
