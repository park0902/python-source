import pickle

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    print(network)


init_network()
