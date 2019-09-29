
import pickle

def writer(obj,name):
    with open(name + '.pickle', 'wb') as openfile :
        pickle.dump(obj, openfile)

def reader(name):
    with open(name + '.pickle', 'rb') as openfile :
        return pickle.load(openfile)
        
