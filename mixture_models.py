from __future__ import division
import warnings
import numpy as np
#import scipy as sp
from matplotlib import image
from random import randint
#from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)

def getInitialMeans(values,k,dim):
    rArray=[]
    for i in range(len(dim)-1):
        r = np.random.randint(0,dim[i],k)
        rArray.append(r)
    l = [ values[rArray[0][i]][rArray[1][i]] for i in range(k) ]
    return np.array(l) 

def getDistances(values,points,k):
    kArray = []
    for i in range(k):
        tmp = np.subtract(values,points[i])
        tmp = np.square(tmp)
        tmp = np.sum(tmp,1)
        tmp = np.sqrt(tmp)
        kArray.append(tmp)
    return np.array(kArray)

    

def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    #1. If initial is None, create a random initial point list from data
    dim = [np.size(image_values,0),np.size(image_values,1),np.size(image_values,2)]
    if initial_means == None:
        initial_means = getInitialMeans(image_values,k,dim)
        
    #2. Loop through initial list and subtract it from the data points
    sz = 1
    for i in range(len(dim)-1): sz = sz * dim[i]
    arr_reshaped = np.reshape(image_values,(sz,dim[len(dim)-1]))

    #3. Square sum and root results to get distance from eack k point
    kArray = getDistances(arr_reshaped,initial_means,k)
    
    #4. Build array containing dataset for each k
    minValues = np.min(kArray,0)
    kValues = []
    kIndexes = []
    kAvg = []
    for i in range(k):
        tmp = np.where(kArray[i] <= minValues)
        kIndexes.append(tmp)
        kValues.append(arr_reshaped[tmp])
    np.array(kValues)
    for i in range(k):
        arr = kValues[i]
        avg = np.mean(arr,0)
        kAvg.append(avg)   
    kAvg = np.array(kAvg)

    #5. Test for convergence and compile new image data to return
    convergence=False
    convArray = np.subtract(initial_means,kAvg)
    convArray = np.absolute(convArray)
    for row in convArray:
        for item in row:
            if item < 1e-6:
                convergence=True
            else:
                convergence=False
                break
        if convergence==False: break
    if convergence:
        arr_orig = np.ndarray(shape=(sz,dim[len(dim)-1]))
        for i in range(k):
            arr_orig[kIndexes[i]] = kAvg[i]

        arr_orig = np.reshape(arr_orig,(dim[0],dim[1],dim[2]))
        return arr_orig
    
    #6. If no convergence, get new mean for each array cluster and recurse
    else:
        return k_means_cluster(image_values,k,kAvg)


def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if means is None:
            self.means = [0] * num_components
        else:
            self.means = means
        self.variances = [0] * num_components
        self.mixing_coefficients = [0] * num_components

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        # TODO: finish this
        raise NotImplementedError()

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        # TODO: finish this
        raise NotImplementedError()

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        # TODO: finish this
        raise NotImplementedError()

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        raise NotImplementedError()

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        # TODO: finish this
        raise NotImplementedError()

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        raise NotImplementedError()


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        # TODO: finish this
        raise NotImplementedError()


def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    raise NotImplementedError()


class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function
        raise NotImplementedError()


def bayes_info_criterion(gmm):
    # TODO: finish this function
    raise NotImplementedError()


def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    """
    # TODO: finish this method
    raise NotImplementedError()


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    raise NotImplementedError()
    bic = 0
    likelihood = 0
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs
