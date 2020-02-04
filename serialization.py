import pickle

def serialize(data, path):
    outfile = open(path, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

def deserialize(path):
    return pickle.load(open(path, "rb"))
