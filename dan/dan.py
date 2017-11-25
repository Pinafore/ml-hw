from numpy import *
from numpy import allclose
import pickle, time, argparse
from collections import Counter

kINIT_WIDTH = 0.08

class Adagrad():

    def __init__(self, dim, lr):
        self.dim = dim
        self.eps = 1e-3

        # initial learning rate
        self.learning_rate = lr

        # stores sum of squared gradients
        self.h = zeros(self.dim)

    def rescale_update(self, gradient):
        curr_rate = zeros(self.h.shape)
        self.h += gradient ** 2
        curr_rate = self.learning_rate / (sqrt(self.h) + self.eps)
        return curr_rate * gradient

    def reset_weights(self):
        self.h = zeros(self.dim)

def softmax(w):
    ew = exp(w - max(w))
    return ew / sum(ew)

def relu(x):
    return x * (x > 0)

def drelu(x):
    return x > 0

def crossent(label, classification):
    return -sum(label * log(classification))

def dcrossent(label, classification):
    return classification - label


def unroll_params(arr, d, dh, len_voc, deep=1, labels=2, wv=True):

    mat_size = dh * dh
    ind = 0

    params = []
    if deep > 0:
        params.append(arr[ind : ind + d * dh].reshape( (dh, d) ))
        ind += d * dh
        params.append(arr[ind : ind + dh].reshape( (dh, ) ))
        ind += dh

        for i in range(1, deep):
            params.append(arr[ind : ind + mat_size].reshape( (dh, dh) ))
            ind += mat_size
            params.append(arr[ind : ind + dh].reshape( (dh, ) ))
            ind += dh

    params.append(arr[ind: ind + labels * dh].reshape( (labels, dh)))
    ind += dh * labels
    params.append(arr[ind: ind + labels].reshape( (labels, )))
    ind += labels
    if wv:
        params.append(arr[ind : ind + len_voc * d].reshape( (d, len_voc)))
    return params

# roll all parameters into a single vector
def roll_params(params):
    return concatenate( [p.ravel() for p in params])

class DeepAveragingNetwork:
    def __init__(self, depth, hidden_dimension, word_dimension,
                 num_labels, vocab_size, rho,
                 nonlinearity=relu, nonlinearity_grad=drelu,
                 update_wv=True):
        self._update_wv = update_wv
        self._vocab_size = vocab_size
        self._depth = depth
        self._rho = rho
        self._word_dimension = word_dimension
        self._hidden_dimension = hidden_dimension
        self._num_labels = num_labels
        self._params = self.init_params()
        self._f = nonlinearity
        self._df = nonlinearity_grad

    def update_params(self, params):
        """
        Given the rolled representation of the parameters, save the
        unrolled version as our current set of parameters.
        """

        self._params = unroll_params(params, self._word_dimension,
                                     self._hidden_dimension, self._vocab_size,
                                     deep=self._depth, labels=self._num_labels)

    def init_params(self):
        params = []
        if self._depth > 0:
            # First parameters from word embedding
            params.append((random.rand(self._hidden_dimension,
                                       self._word_dimension) * 2 - 1) *
                           kINIT_WIDTH)
            # First layer bias
            params.append((random.rand(self._hidden_dimension, ) * 2 - 1) *
                           kINIT_WIDTH)

            for i in range(1, self._depth):
                # Layer from hidden layer to hidden layer
                params.append((random.rand(self._hidden_dimension,
                                           self._hidden_dimension) * 2 - 1)
                              * kINIT_WIDTH)
                params.append((random.rand(self._hidden_dimension, ) * 2 - 1) *
                              kINIT_WIDTH)

        # Final softmax classification layer
        params.append((random.rand(self._num_labels,
                                   self._hidden_dimension) * 2 - 1)
                      * kINIT_WIDTH)
        params.append((random.rand(self._num_labels, ) * 2 - 1) * kINIT_WIDTH)

        print('randomly initializing word embeddings...')
        orig_We = (random.rand(self._hidden_dimension, self._vocab_size) * 2 - 1) * kINIT_WIDTH

        # add We matrix to params
        params += (orig_We, )
        
        return params

    def init_grads(self):

        grads = []
        if self._depth > 0:
            grads.append(zeros((self._hidden_dimension, self._word_dimension)))
            grads.append(zeros(self._hidden_dimension, ))

            for i in range(1, self._depth):
                grads.append(zeros( (self._hidden_dimension,
                                     self._hidden_dimension) ))
                grads.append(zeros( (self._hidden_dimension, ) ))

        grads.append(zeros( (self._num_labels, self._hidden_dimension) ))
        grads.append(zeros( (self._num_labels, ) ))
        if self._update_wv:
            grads.append(zeros((self._word_dimension, len_voc)))
        return grads

    def get_w(self, level):
        return self._params[level * 2]

    def get_b(self, level):
        return self._params[level * 2 + 1]

    def get_embed(self):
        return self._params[-1]

    def activations(self, sentence):
        """
        Create matrix of the activations for each level
        """

        # This is the matrix you'll need to return initialized to zero
        activations = zeros((self._depth, self._hidden_dimension))

        # This is the average of the embeddings of the sentence
        av = average(self.get_embed()[:, sentence], axis=1)

        # forward prop

        return activations

    def prediction(self, sentence, activation=None):
        Ws = self.get_w(self._depth)
        bs = self.get_b(self._depth)

        if self._depth == 0:
            av = average(self._get_embed()[:, sentence], axis=1)
            pred = softmax(Ws.dot(av) + bs).ravel()
        else:
            if activation is None:
                activation = self.activations(sentence)
            pred = softmax(Ws.dot(activation[-1]) + bs).ravel()
        return pred

    # does both forward and backprop
    def objective_and_grad(self, data, fine_tune=True):

        params = self._params
        grads = self.init_grads()
        error_sum = 0.0

        for sent,label in data:

            if len(sent) == 0:
                continue

            # We want the prediction to match this vector
            target = zeros(self._num_labels)
            target[label] = 1.0

            # input is average of all nouns in sentence
            curr_sent = sent

            av = average(params[-1][:, curr_sent], axis=1)
            acts = self.activations(curr_sent)

            # compute softmax error
            Ws = self.get_w(self._depth)
            bs = self.get_b(self._depth)

            pred = self.prediction(curr_sent)
            error_sum += crossent(target, pred)
            soft_delta = dcrossent(target, pred)

            # TODO: Implement backpropagation error!


                # backprop 



        # Add the contribution of the regularization to the gradient

        cost = error_sum / len(data) 

        return cost, grad


    def validate(self, data, fold, f=relu):

        correct = 0.
        total = 0.

        for sent, label in data:

            if len(sent) == 0:
                continue

            av = average(self._params[-1][:, sent], axis=1)

            pred = self.prediction(sent)

            if argmax(pred) == label:
                correct += 1

            total += 1

            print('accuracy on ', fold, correct, total, str(correct / total),
                  '\n')
            return correct / float(total)



if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='sentiment DAN')
    parser.add_argument('-data', help='location of dataset',
                        default='../data/sentiment/')
    parser.add_argument('-vocab', help='location of vocab',
                        default='../data/sentiment/vocab.pkl')
    parser.add_argument('-d', help='word embedding dimension',
                        type=int, default=300)
    parser.add_argument('-dh', help='hidden dimension', type=int, default=300)
    parser.add_argument('-deep', help='number of hidden layers',
                        type=int, default=3)
    parser.add_argument('-rho', help='regularization weight',
                        type=float, default=1e-4)
    parser.add_argument('-labels', help='number of labels', type=int, default=5)
    parser.add_argument('-ft', help='fine tune word vectors',
                        type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size',
                        type=int, default=15)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs',
                        type=int, default=5)
    parser.add_argument('-agr', '--adagrad_reset',
                        help='reset  sum of squared gradients after this many \
                         epochs', type=int, default=50)
    parser.add_argument('-lr', help='adagrad initial learning rate',
                        type=float, default=0.005)
    parser.add_argument('-o', '--output', help='location of output model', \
                         default='sentiment_params.pkl')

    args = parser.parse_args()
    d = args.d
    dh = args.dh


    # load data
    train = pickle.load(open(args.data+'train.pkl', 'rb'))
    test = pickle.load(open(args.data+'test.pkl', 'rb'))
    vocab = pickle.load(open(args.vocab, 'rb'))
    len_voc = len(vocab)

    dan = DeepAveragingNetwork(args.deep, args.dh, args.d, args.labels,
                               len_voc, args.ft)

    for split in [train, test]:
        c = Counter()
        tot = 0
        for sent, label in split:
            c[label] += 1
            tot += 1
        print(split, c, tot)

    params = dan._params
    r = roll_params(params)

    # output log and parameter file destinations
    param_file = args.output
    log_file = param_file.split('_')[0] + '_log'
    
    dim = r.shape[0]
    print('parameter vector dimensionality:', dim)

    log_object = open(log_file, 'w')

    # minibatch adagrad training
    ag = Adagrad(r.shape, args.lr)
    min_error = float('inf')

    for epoch in range(0, args.num_epochs):

        lstring = ''

        # create mini-batches
        random.shuffle(train)
        batches = [train[x : x + args.batch_size] for x in range(0, len(train),
                   args.batch_size)]

        epoch_error = 0.0
        ep_t = time.time()
        for batch_ind, batch in enumerate(batches):
            now = time.time()
            err, grad = dan.objective_and_grad(batch, fine_tune=args.ft)

            update = ag.rescale_update(grad)
            r = r - update
            dan.update_params(r)
            lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + \
                    ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
            log_object.write(lstring + '\n')
            log_object.flush()
            epoch_error += err

        # done with epoch
        print(time.time() - ep_t)
        print('done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error)
        lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                 + ' min error = ' + str(min_error) + '\n'
        log_object.write(lstring)
        log_object.flush()

        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            params = unroll_params(r, d, dh, len_voc, deep = args.deep, labels=args.labels)
            # d_score = validate(dev, 'dev', params, args.deep)
            pickle.dump( params, open(param_file, 'wb'))

        log_object.flush()

        # reset adagrad weights
        if epoch % args.adagrad_reset == 0 and epoch != 0:
            ag.reset_weights()

    log_object.close()

    # compute test score
    params = unroll_params(r, d, dh, len_voc, deep = args.deep, labels=args.labels)
    t_score = validate(test, 'test', params, args.deep)
