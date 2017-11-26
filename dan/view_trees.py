import pickle

if __name__ == "__main__":
    rev_vocab = pickle.load(open("../data/sentiment/vocab.pkl", 'rb'))
    train = pickle.load(open("../data/sentiment/train.pkl", 'rb'))
    test = pickle.load(open("../data/sentiment/test.pkl", 'rb'))

    vocab = {}
    for ii in rev_vocab:
        vocab[rev_vocab[ii]] = ii
    
    for dd in [train, test]:
        for ii in dd:
            print("%i\t%s" % (ii[1], " ".join(vocab[x] for x in ii[0])))
