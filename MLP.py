from numpy import *
from random import gauss
import scipy.io
import pylab
import math
import time
from scipy.misc import imresize
from exec_time import *

eta = 0.1 # the learning rate
miu = 0.5  # the momentum term
MAX_NUM_ITERATIONS = 100

class Layer(object):
    
    def __init__(self, num_units, prev_num_units):
		self.num_units = num_units
		self.prev_num_units = prev_num_units
		self.weights = self.init_weights()
		self.prev_delta_weights = self.reset_delta_weights()
		self.bias = self.init_bias()
		self.residuals = array([])
		self.transfers = array([])	

    def init_weights(self):
        raise NotImplementedError()

    def reset_delta_weights(self):
        raise NotImplementedError()

    def init_bias(self):
        raise NotImplementedError()
    
    def compute_activations(self, inputs_prev_layer):
        raise NotImplementedError()
        
    def compute_residuals(self, wr):
        raise NotImplementedError()
    
    """function used to compute the transfers for the hidden layers"""
    def sigma(self, x):
        if x < 0:
            exponential = math.e ** (x)
            result = exponential / (1 + exponential)
        else:
            result = 1/(1 + math.e ** (-x))
        return result

    def compute_transfers(self):
        raise NotImplementedError()
        
    def forward_propagation(self, inputs_prev_layer):
        self.compute_activations(inputs_prev_layer)
        self.compute_transfers()
        return self.transfers

    def compute_Wr(self):
        return array([i for i in transpose(self.weights).dot(self.residuals) for j in [0,1]])
    
    def back_propagation(self, next_layer_Wr):
        """Remark! for the output layer the next_layer_Wr argument is the label"""
        self.compute_residuals(next_layer_Wr)
        return self.compute_Wr()   
        
    def update_weights(self, inputs_prev_layer):
        delta_w = -eta*(1-miu)*transpose(array([self.residuals])).dot(array([inputs_prev_layer]))+ miu * self.prev_delta_weights
        self.weights += delta_w
        self.prev_delta_weights = delta_w
        self.bias -= self.residuals

class HiddenLayer(Layer):
    
    def __init__(self, num_units, prev_num_units):
        super(HiddenLayer, self).__init__(num_units, prev_num_units)   
        self.odd_activations = array([])
        self.sigma_even_activations = array([])      
    
    def init_weights(self):
        weights = []
        sigma = 1/math.sqrt(self.prev_num_units)
        for i in range(2*self.num_units):
            weights.append([])
            for j in range(self.prev_num_units):
                weights[i].append(gauss(0, sigma))
        return array(weights)
        
    def reset_delta_weights(self):
        return zeros((self.num_units*2, self.prev_num_units))
        
    def init_bias(self):
        bias = []
        sigma = 1/math.sqrt(self.prev_num_units)
        for i in range(2*self.num_units):
            bias.append(gauss(0, sigma))   
        return array(bias)

    def compute_activations(self, inputs_prev_layer):
        activations = self.weights.dot(inputs_prev_layer) + self.bias
        self.odd_activations = activations[range(0,self.num_units *2, 2)]
        self.sigma_even_activations = array(map(self.sigma, activations[range(1,self.num_units *2, 2)]))
            
    def compute_transfers(self): 
        self.transfers = self.odd_activations*self.sigma_even_activations

    def compute_residuals(self, wr):
        derivative = array([self.sigma_even_activations[j] * [1, self.odd_activations[j]*(1 -\
        self.sigma_even_activations[j])][i] for j in range(self.num_units) for i in [0, 1]])
        self.residuals = wr*derivative

class OutputLayer(Layer):
    
    def __init__(self, prev_num_units):
        super(OutputLayer, self).__init__(1, prev_num_units)
        self.activation = None

        
    def init_weights(self):
        weights = []
        sigma = 1/math.sqrt(self.prev_num_units)
        i = 0
        weights.append([])
        for j in range(self.prev_num_units):
            weights[i].append(gauss(0, sigma))
        return array(weights)
        
    def reset_delta_weights(self):
        return zeros((1, self.prev_num_units))
        
    def init_bias(self):
        sigma = 1/math.sqrt(self.prev_num_units)
        bias = gauss(0, sigma)   
        return bias

    def compute_activations(self, inputs_prev_layer):
        self.activation = (self.weights.dot(inputs_prev_layer) + self.bias)[0]
        
    def compute_transfers(self):
        self.transfers = math.copysign(1, self.activation)

    def compute_residuals(self, label):
        self.residuals = -label*self.sigma(-label*self.activation)


class MLP(object):
    
    def __init__(self, num_units_per_layer, training_set, validation_set, test_set):
        self.n = len(training_set['Xtrain'])
        self.training_data = training_set['Xtrain']
        self.training_labels = training_set['Ytrain']
        self.validation_data = validation_set['Xtrain']
        self.validation_labels = validation_set['Ytrain']
        self.test_data = test_set['Xtest']
        self.test_labels = test_set['Ytest']
        self.input_dimension = len(self.training_data[0])
        self.num_hidden_layers = len(num_units_per_layer)
        self.layers = []
        if self.num_hidden_layers:
            self.layers.append( HiddenLayer(num_units_per_layer[0], self.input_dimension) )
        for i in range(1, self.num_hidden_layers):
            self.layers.append( HiddenLayer(num_units_per_layer[i], num_units_per_layer[i-1]) )
        self.layers.append( OutputLayer(num_units_per_layer[-1]) )

    def training_step(self):
        for d in range(len(self.training_data)):
            forward_output = self.training_data[d]
            for l in self.layers:
                forward_output = l.forward_propagation(forward_output)
            
            backward_output = self.training_labels[d]
            for l in self.layers.__reversed__():
                backward_output = l.back_propagation(backward_output)

            self.layers[0].update_weights(self.training_data[d])
            for i in range(1, self.num_hidden_layers):
                self.layers[i].update_weights(self.layers[i-1].transfers)

    def train(self):
        errors = []
        iteration = 0
        while(True): 
            iteration += 1
            self.training_step()
            errors.append([self.compute_error(1, 1), self.compute_error(0, 1), self.compute_error(0, 0)])
            #stop conditions
            if iteration > 20 and sum(errors[-20:-11][1]) < sum(errors[-10:][1]) + 0.01: break
            if iteration == MAX_NUM_ITERATIONS: break
        lines = pylab.plot(range(iteration), errors)
        pylab.xlabel("Number of iterations")
        pylab.ylabel("Errors for dataset " + dataset)
        pylab.figlegend(lines, ("training-logistic","validation-logistic", "validation-0/1"), "upper right")
        pylab.savefig("fig" + timestamp + ".pdf")        
        #self.plot_train(iteration, errors) # used for experiments
        #self.plot_validation(iteration, errors)
            
    def plot_train(self, nb_iterations, errors):
        x = arange(nb_iterations)
        pylab.plot( x, array(errors).T[0], '-', label='$\eta$ = ' + str(eta) )
        pylab.xlabel("Number of iterations")
        pylab.ylabel("Training error for dataset " + dataset)
        pylab.legend()
        pylab.savefig("fig" + timestamp + ".pdf")
    
    def plot_validation(self, nb_iterations, errors):
        x = arange(nb_iterations)
        pylab.plot( x, array(errors).T[1], '-', label='hidden layer: ' + str(hidden_layers) )
        pylab.xlabel("Number of iterations")
        pylab.ylabel("Validation error for dataset " + dataset)
        pylab.legend()
        pylab.savefig("fig" + timestamp + ".pdf")
    
    def compute_error(self, dataset_type, error_type):
        """ dataset_type : 1 for training, 0 for validation, 2 - for test
            error_type - 1 means logistic error, 0 - is for 0/1 error"""
        if dataset_type == 1:
            dataset = self.training_data
            labels = self.training_labels
        elif dataset_type == 0:
            dataset = self.validation_data
            labels = self.validation_labels
        elif dataset_type == 2:
            dataset = self.test_data
            labels = self.test_labels
        else:
            raise Exception
        error = 0.0
        for d in range(len(dataset)):
            forward_output = dataset[d]
            for l in self.layers:
                forward_output = l.forward_propagation(forward_output)
            if error_type == 1:
                error += self.error_function(self.layers[-1].activation, labels[d][0])
            else:
                error += self.is_wrong(self.layers[-1].activation, labels[d][0])
        error /= float(len(dataset)) 
        return error

    """ logistic error function"""
    def error_function(self, output, label):
        x = label * output
        if x > 0:
            return math.log(1 + math.e ** (-x))
        else:
            return -x + math.log(1 + math.e ** x)
    
    def is_wrong(self, output, label):
        return (label*(output - 0.5) <= 0)
        
def find_alpha_min_max(dataset):
    global alpha_max, alpha_min
    alpha_max = float(amax(dataset))
    alpha_min = float(amin(dataset))

def normalize(dataset):
    dataset = (dataset - alpha_min)/(alpha_max - alpha_min)
    return dataset

""" show the image with index k"""
def showImage(data, k):
    img1 = []
    img1 = data[k].reshape((28,28))
    pylab.imshow( transpose(array(img1)))
    pylab.show()

def showRandomImages():
    d = scipy.io.loadmat('training_3-5.mat') # corresponding MAT file
    data = d['Xtrain']
    labels = d['Ytrain']
    n = 5
    print "Show %d random images: " % (n)
    for i in random.randint(0, len(data), n):
        print labels[i]
        showImage(data, i)

""" function used to test the classifier with various parameters:
    learning rate, momentum term, hidden layers with units"""
def test_MLP():
    global dataset
    dataset = "4-9"#"3-5"#
    global timestamp
    timestamp = str(int(time.time())) #to be used when creating an output file
    
    """ read and normalize data """
    d = scipy.io.loadmat('training_'+dataset+'.mat') # training dataset
    validation = scipy.io.loadmat('validation_'+dataset+'.mat') # validation
    test = scipy.io.loadmat('mp_'+dataset+'_data.mat') # testing
    find_alpha_min_max(concatenate([d['Xtrain'], validation['Xtrain']]) )
    d['Xtrain'] = normalize(d['Xtrain'])
    validation['Xtrain'] =  normalize(validation['Xtrain'])
    test['Xtest'] = normalize(test['Xtest'])
    
    """ create MLP classifier """
    hidden_layers = [25]
    classifier = MLP(hidden_layers, d, validation, test)
    ts = time.time()
    classifier.train()
    te = time.time()
    train_err = classifier.compute_error(1, 0)
    validation_err = classifier.compute_error(0, 0)
    test_err = classifier.compute_error(2, 0)
    print "%f\t%f\t%f\t%2.2f\n" % (train_err, validation_err, test_err, te-ts) 

if __name__ == "__main__":
    test_MLP()
