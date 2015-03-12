from numpy import *
from scipy.stats import norm
import scipy.io
import math
import time
import random
from scipy.misc import imresize
from exec_time import *

class SVM(object):
    K = None
    alpha = None
    f = None
    I_low = None
    I_up = None
    b = None

    def __init__(self, C, TAU, K, training_set_labels, K_test, test_set_labels, cross_validation=False):
        self.C = C
        self.TAU = TAU
        self.K = K # precomputed kernel of training dataset
        self.t = training_set_labels # training labels
        self.K_test = K_test
        self.test_labels = test_set_labels
        self.cross_validation = cross_validation
       
    """ SMO criterion function """   
    def phi(self):
        alpha_t = array([self.alpha*self.t])
        return alpha_t.dot(self.K).dot(alpha_t.T)/2 - sum(self.alpha)

    """ SMO algorithm """
    @timeit
    def train(self):
        self.alpha = zeros(len(self.t))
        self.f = -1.0*self.t[:]
        self.I_low = set([i for i in range(len(self.alpha)) if self.t[i] == -1])
        self.I_up = set([i for i in range(len(self.alpha)) if self.t[i] == +1])
        step = 0 # the step of the SMO algorithm
        to_plot = []
        while(True):
            (i, j) = self.select_pair()
            if j == -1: # no violated pair found
                break #stop algorithm
            old_alpha_i = self.alpha[i]
            old_alpha_j = self.alpha[j]
            sigma = self.t[i] * self.t[j]
            #Compute L, H from (5)
            w = old_alpha_i + sigma * old_alpha_j
            L = max(0.0, sigma*w - (sigma == 1) * self.C )
            H = min(self.C, sigma*w + (sigma == -1) * self.C)
            eta = self.K[i,i] + self.K[j,j] - 2*self.K[i,j]
            #Update alpha[j]
            if eta > 1E-15:
                #Compute the minimum along the direction of the constraint from (6)
                self.alpha[j] = old_alpha_j + self.t[j] * (self.f[i] - self.f[j]) / eta
                #Clip unconstrained minimum to the ends of the line segment according to (7)
                if self.alpha[j] > H:
                    self.alpha[j] = H
                if self.alpha[j] < L:
                    self.alpha[j] = L
            else: # the second derivative is close to zero
            #Compute phi_H and phi_L accoding to (8)
                L_i = w - sigma*L
                H_i = w - sigma*H
                v_i = self.f[i] + self.t[i] - old_alpha_i*self.t[i]*self.K[i, i] - old_alpha_j*self.t[j]*self.K[i, j] 
                v_j = self.f[j] + self.t[j] - old_alpha_i*self.t[i]*self.K[i, j] - old_alpha_j*self.t[j]*self.K[j, j] 
                phi_L = 1/2*(self.K[i, i]*(L_i*L_i) + self.K[j, j]*L*L) \
                 + sigma*self.K[i, j]*L_i*L + self.t[i]*L_i*v_i + self.t[j]*L*v_j - L_i - L
                phi_H = 1/2*(self.K[i, i]*(H_i*H_i) + self.K[j, j]*H*H) \
                 + sigma*self.K[i, j]*H_i*H + self.t[i]*H_i*v_i + self.t[j]*H*v_j - H_i - H
                if phi_L > phi_H:
                    self.alpha[j] = H
                else:
                    self.alpha[j] = L
            self.alpha[i] = w - sigma*self.alpha[j] # compute new alpha[i] from the new alpha[j]
            if (not self.cross_validation) and (step % 20 == 0):
                to_plot.append( (step, self.phi(), self.f[i] - self.f[j]) )       
            self.f += self.t[i]*(self.alpha[i] - old_alpha_i)*self.K[:,i] + self.t[j]*(self.alpha[j] - old_alpha_j)*self.K[:,j]
            self.update_I_sets(i)
            self.update_I_sets(j)
            step += 1
        #compute the bias term b
        I_0 = [i for i in range(len(self.alpha))
                    if ((0 < self.alpha[i]) and (self.alpha[i] < self.C))]
        y = (self.alpha*self.t).dot(self.K[:,I_0])
        if len(I_0) == 0:
            self.b = 0
        else:
            self.b = -1.0*sum(self.t[I_0] - y)/len(I_0)
        #write in a file SMO_criterion computed values for plotting
        if not self.cross_validation:
            savetxt ('SMO_criterion', to_plot, fmt='%.4f')

    """ update I_up and I_low sets 
        according to the modified index (i) in alpha vector"""
    def update_I_sets(self, i):
        if( (0 < self.alpha[i]) and (self.alpha[i] < self.C) ):
            self.I_up.add(i)
            self.I_low.add(i)
        elif( ((self.t[i] == 1) and (self.alpha[i] == 0)) or
            ((self.t[i] == -1) and (self.alpha[i] == self.C)) ):
            self.I_up.add(i)
            self.I_low.discard(i)
        else:
            self.I_low.add(i)
            self.I_up.discard(i)

    """ find the most violated pair"""
    def select_pair(self):
        i_up = min([(i, self.f[i]) for i in self.I_up], key=lambda x: x[1])[0]
        i_low = max([(i, self.f[i]) for i in self.I_low], key=lambda x: x[1])[0]
        if self.f[i_low] <= self.f[i_up] + 2 * self.TAU:
            i_low = -1
            i_up = -1
        return (i_low , i_up )
    
    """ compute the 0/1 error on the training dataset"""
    @timeit
    def train_error(self):
        y = (self.alpha*self.t).dot(self.K) - self.b
        classified = array(map(lambda x: math.copysign(1,x), y))*self.t
        error = float(len(classified[classified<=0])) / len(classified)
        return error

    """ compute the 0/1 error on the test dataset"""        
    @timeit
    def test_error(self):
        y = (self.alpha*self.t).dot(self.K_test) - self.b
        classified = array(map(lambda x: math.copysign(1,x), y))*self.test_labels
        error = float(len(classified[classified<=0])) / len(classified)
        return error
    
class Cross_validation:
    
    def __init__(self, training_labels, A_train, TAU = 1E-1):
        self.A_train = A_train
        self.TAU = TAU
        self.t = training_labels # training labels

    """ M-fold cross validation to asses classification error for given C
        and Gauss_param (which is included in the precomputed kernel K) """
    @timeit
    def cross_validation(self, M, C, K):
        len_training = size(self.A_train, 0)
        indexes = range(len_training)
        random.shuffle(indexes)
        test_error_avg = 0.0
        train_error_avg = 0.0
        for i in range(M):
            train_indexes = indexes[:i*len_training//M]+indexes[(i+1)*len_training//M:]
            test_indexes = indexes[i*len_training//M:(i+1)*len_training//M]
            K_test = K[ train_indexes, :][:, test_indexes ]
            test_labels = self.t[ test_indexes ]
            K_train = K[ train_indexes, :][: ,train_indexes ]
            training_labels = self.t[ train_indexes ]
            classifier = SVM(C, self.TAU, K_train, training_labels, K_test, test_labels, True)
            classifier.train()
            test_error = classifier.test_error()
            train_error = classifier.train_error()
            test_error_avg += test_error
            train_error_avg += train_error
        return test_error_avg/M
    
    """ find the best combination (with the minimum error) of parameters
        among a set of selected sensible values """
    @timeit
    def choose_best_SVM_parameters(self):
        Folds = 10
        C = 2**arange(4.0, 5.0, 1.0)
        Gauss_param =2**arange(-7.0, -2.0, 0.5)
        best_err = 1.0
        for g in Gauss_param:
            K = exp(-g * self.A_train)
            for c in C:
                err = self.cross_validation(Folds, c, K)
                print "%f %f %f" % (c, g, err)
                with open('SVM_cv_errors_6000p_28x28_14Dec.txt','a') as f:    
                    f.write("%f\t%f\t%f\n" % (c, g, err) )
                if err < best_err:
                    best_err = err
                    best_c = c
                    best_g = g
        return (best_c, best_g, best_err)

""" find the minimum and the maximum value from the given dataset
    and store them in global variables to be used for normalization"""   
def find_alpha_min_max(dataset):
    global alpha_max, alpha_min
    alpha_max = float(amax(dataset))
    alpha_min = float(amin(dataset))

""" normalize the dataset between 0 and 1 """
@timeit
def normalize(dataset):
    dataset = (dataset - alpha_min)/(alpha_max - alpha_min)
    return dataset

""" preprocess data by normalizing the values"""
def load_raw_data(dataset):
    data = scipy.io.loadmat('mp_'+dataset+'_data.mat') # corresponding MAT file
    find_alpha_min_max(data['Xtrain'])
    d = {}
    test = {}
    d['Xtrain'] = normalize(data['Xtrain'])
    d['Ytrain'] = data['Ytrain'].flatten()
    test['Xtest'] = normalize(data['Xtest'])
    test['Ytest'] = data['Ytest'].flatten()
    return (d, test)

""" * precompute the squared distance matrices A for train and for test
    * save them into files in order to speed-up the algorithm"""
@timeit
def compute_squared_dist():
    outfile_train = 'squared_dist_train.npy'
    outfile_test = 'squared_dist_test.npy'
    dataset = "4-9"

    train, test = load_raw_data(dataset)
    xtrain = train['Xtrain']
    ztest = test['Xtest']

    train_len = len(xtrain)
    x_xT = xtrain.dot(xtrain.T)
    dx = diag(diag(x_xT))
    ones_n = ones( (train_len, train_len) )
    A_train = ( dx.dot(ones_n) + ones_n.dot(dx) )/2 - x_xT

    test_len = len(ztest)
    x_zT = xtrain.dot(ztest.T)
    dz = diag(sum(ztest**2, axis=-1))
    ones_n = ones( (train_len, test_len) )
    A_test = ( dx.dot(ones_n) + ones_n.dot(dz) )/2 - x_zT

    save(outfile_train, A_train)
    save(outfile_test, A_test)

""" run cross validation"""
@timeit
def assess_parameters():
    dataset = "4-9"
    d, test = load_raw_data(dataset)
    infile_train = 'squared_dist_train.npy'
    A_train = load(infile_train, mmap_mode=None)

    cv = Cross_validation(d['Ytrain'], A_train)
    res = cv.choose_best_SVM_parameters()
    with open('SVM_bestparams.txt','a') as f:    
        f.write("C %f, Gauss_param %f, avg_error %f\n" % (res[0],res[1],res[2]) )

""" run SVM classifier """
def classification():
    dataset = "4-9"
    C = 2**4
    TAU = 1E-1
    GAUSS_PARAM = 2**(-4)
    
    infile_train = 'squared_dist_train.npy'
    A_train = load(infile_train, mmap_mode=None)
    infile_test = 'squared_dist_test.npy'
    A_test = load(infile_test, mmap_mode=None)

    d, test = load_raw_data(dataset)

    K_train = exp(-GAUSS_PARAM * A_train)
    K_test = exp(-GAUSS_PARAM * A_test)
    
    classifier = SVM(C, TAU, K_train, d['Ytrain'], K_test, test['Ytest'])
    classifier.train()
    train_error = classifier.train_error()
    test_error =  classifier.test_error()
    print "With C:%f G:%f: Error on train: %f; Error on test: %f\n" %(C, GAUSS_PARAM, train_error, test_error)

def main():
    classification() #run SVM classification
    #assess_parameters() #run cross-validation

if __name__ == "__main__":
    print " You should call first compute_squared_dist(), it runs for several hours "
    #compute_squared_dist() #precompute A matrices used for kernel calculation
    #main()
    
    
