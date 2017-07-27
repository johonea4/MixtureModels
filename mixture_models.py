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

def getMeans(values, kArray,k):
    minValues = np.min(kArray,0)
    kValues = []
    kIndexes = []
    kAvg = []
    for i in range(k):
        tmp = np.where(kArray[i] <= minValues)
        kIndexes.append(tmp)
        kValues.append(values[tmp])
        kAvg.append(np.mean(values[tmp],0))
    np.array(kValues)
    kAvg = np.array(kAvg)
    return (kValues,kIndexes,kAvg)

def getConvergence(kMeans, initial_means):
    convergence=False
    convArray = np.subtract(initial_means,kMeans)
    convArray = np.absolute(convArray)
    for row in convArray:
        for item in row:
            if item < 1e-6:
                convergence=True
            else:
                convergence=False
                break
        if convergence==False: break
    return convergence

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
#    raise NotImplementedError()

    #1. If initial is None, create a random initial point list from data
    dim = [np.size(image_values,0),np.size(image_values,1),np.size(image_values,2)]
    if initial_means == None:
        initial_means = getInitialMeans(image_values,k,dim)
        
    #2. Loop through initial list and subtract it from the data points
    sz = 1
    for i in range(len(dim)-1): sz = sz * dim[i]
    arr_reshaped = flatten_image_matrix(image_values) #np.reshape(image_values,(sz,dim[len(dim)-1]))

    #3. Square sum and root results to get distance from eack k point
    kArray = getDistances(arr_reshaped,initial_means,k)
    
    #4. Build array containing dataset for each k
    kImage,kIndexes,kMeans = getMeans(arr_reshaped,kArray,k)
    #5. Test for convergence and compile new image data to return
    while(getConvergence(kMeans,initial_means)==False):
        initial_means = kMeans
        kArray = getDistances(arr_reshaped,initial_means,k)
        kImage,kIndexes,kMeans=getMeans(arr_reshaped,kArray,k)
    
    arr_orig = np.ndarray(shape=(sz,dim[len(dim)-1]))
    for i in range(k):
        arr_orig[kIndexes[i]] = kMeans[i]

    arr_orig = unflatten_image_matrix(arr_orig,dim[0])
    return arr_orig
    
    #6. If no convergence, get new mean for each array cluster and recurse
    # else:
    #     return k_means_cluster(image_values,k,kAvg)


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
        #Alright, lets calculate the ln part of the equation
        a1 = np.multiply(self.variances,2*np.pi)
        a1 = np.log(a1)
        a1 = np.multiply(a1,-0.5)
        a2 = np.subtract(val,self.means)
        a2 = np.square(a2)
        a3 = np.multiply(2,self.variances)
        a2 = np.divide(a2,a3)
        a1 = np.subtract(a1,a2)
        a1 = np.exp(a1)
        a0 = np.multiply(self.mixing_coefficients,a1)

        joint_prob = np.log(np.sum(a0))
        return joint_prob

    def x_prob(self):
        arr = self.image_matrix.flatten()
        px = []

        for i in range(self.num_components):
            a1 = -0.5*np.log(2.0*np.pi*self.variances[i])
            a2 = np.subtract(arr,self.means[i])
            a2 = np.square(a2)
            a2 = np.divide(a2,2.0*self.variances[i])
            a3 = np.subtract(a1,a2)
            a3 = np.exp(a3)
            px.append(a3)
        
        return px

    def x_prob2(self):
        arr = self.image_matrix.flatten()
        px = []

        for i in range(self.num_components):
            a1 = 1.0/np.sqrt(2.0*np.pi*self.variances[i])
            a2 = np.subtract(arr,self.means[i])
            a2 = np.square(a2)
            a2 = np.divide(a2,2.0*self.variances[i])
            a2 = np.multiply(a2,-1.0)
            a2 = np.exp(a2)
            a3 = np.multiply(a1,a2)
            px.append(a3)
        
        return px

    def x_resp(self):
        px = self.x_prob()
        resp = []

        for i in range(self.num_components):
            a1 = px[i]
            a1 = np.multiply(a1,self.mixing_coefficients[i])
            a2 = []
            for j in range(self.num_components):
                b1 = px[j]
                # if(j!=i):
                #     b1 = np.subtract(1.0,b1)
                b1 = np.multiply(b1,self.mixing_coefficients[j])
                if(j==0):
                    a2=b1
                else:
                    a2 = np.add(a2,b1)
            a1 = np.divide(a1,a2)
            resp.append(a1)
        return resp

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
        self.variances = [1] * self.num_components
        arr = self.image_matrix.flatten()
        idx = np.random.randint(0,len(arr),self.num_components)
        self.means = arr[idx]

        val = 1.0/float(self.num_components)
        self.mixing_coefficients = [val] * self.num_components

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

        convergence = False
        pl=self.likelihood()
        arrf = self.image_matrix.flatten()
        nl=pl
        c =0

        while convergence == False:
            #1. Take the initial means and stdev and find the probabilty that they fit
            px = self.x_resp()

            coef = np.mean(px,1)
            self.mixing_coefficients=coef

            m = np.multiply(arrf,px)
            m = np.sum(m,1)
            m = np.divide(m,np.sum(px,1))
            self.means = m

            v = []
            for i in range(self.num_components):
                v.append(np.subtract(arrf,self.means[i]))
            v = np.square(v)
            v = np.multiply(v,px)
            v = np.sum(v,1)
            v = np.divide(v,np.sum(px,1))
            self.variances = v

            nl = self.likelihood()
            c, convergence = convergence_function(pl,nl,c)
            pl=nl

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

        px = self.x_prob()
        indexes = np.argmax(px,0)
        arr = np.array(self.image_matrix.flatten())
        for i in range(self.num_components):
            idx = np.where(indexes==i)
            arr[ idx ] = self.means[i]
        return np.reshape(arr,np.shape(self.image_matrix))

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
        px = self.x_prob()
        px = np.transpose(px)
        px = np.multiply(px,self.mixing_coefficients)
        px = np.sum(px,1)
        px = np.log(px)
        
        l = np.sum(px)
        return l

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
        highest_means = []
        highest_variances = []
        highest_coefficients = []
        highest_segment = []

        likelihood = self.likelihood()

        for i in range(iters):
            self.train_model()
            l = self.likelihood()
            if l > likelihood:
                likelihood = l
                highest_means = self.means
                highest_variances = self.variances
                highest_coefficients = self.mixing_coefficients
                highest_segment = self.segment()
        
        self.means = np.array(highest_means)
        self.variances = np.array(highest_variances)
        self.mixing_coefficients = np.array(highest_coefficients)
        
        return highest_segment

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
        GaussianMixtureModel.initialize_training(self)
        arr = self.image_matrix.flatten()
        converge = False
        while(converge == False):
            init_means = np.array(self.means)
            distances=[]
            for i in range(self.num_components):
                a1 = np.subtract(arr,init_means[i])
                a1 = np.abs(a1)
                distances.append(a1)
            mins = np.min(distances,0)
            final_means = []
            for i in range(self.num_components):
                indexes = np.where(distances[i] <= mins[i])
                a1 = arr[indexes]
                a2 = np.mean(a1)
                final_means.append(a2)
            self.means = np.array(final_means)
            a1 = np.abs(np.subtract(self.means,init_means))
            a2 = np.array([1.0e-6]*self.num_components)
            a1_bool = a1 <= a2
            if np.alltrue(a1_bool):
                converge=True

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
    pMeans = previous_variables[0]
    pVariances = previous_variables[1]
    pCoefficients = previous_variables[2]
    nMeans = new_variables[0]
    nVariances = new_variables[1]
    nCoefficients = new_variables[2]

    diff_means = np.abs(np.subtract(nMeans,pMeans))
    diff_variances = np.abs(np.subtract(nVariances,pVariances))
    diff_coefficients = np.abs(np.subtract(nCoefficients,pCoefficients))

    bool_means = diff_means <= np.abs(np.multiply(pMeans,0.1))
    bool_variances = diff_variances <= np.abs(np.multiply(pVariances,0.1))
    bool_coefficients = diff_coefficients <= np.abs(np.multiply(pCoefficients,0.1))

    if np.alltrue(bool_means) and np.alltrue(bool_variances) and np.alltrue(bool_coefficients):
        conv_ctr = conv_ctr+1
    else:
        conv_ctr = 0
    
    return conv_ctr, conv_ctr > conv_ctr_cap

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        convergence = False
        pl= [ self.means, self.variances, self.mixing_coefficients ]
        arrf = self.image_matrix.flatten()
        nl=pl
        c =0

        while convergence == False:
            #1. Take the initial means and stdev and find the probabilty that they fit
            px = self.x_resp()

            coef = np.mean(px,1)
            self.mixing_coefficients=coef

            m = np.multiply(arrf,px)
            m = np.sum(m,1)
            m = np.divide(m,np.sum(px,1))
            self.means = m

            v = []
            for i in range(self.num_components):
                v.append(np.subtract(arrf,self.means[i]))
            v = np.square(v)
            v = np.multiply(v,px)
            v = np.sum(v,1)
            v = np.divide(v,np.sum(px,1))
            self.variances = v

            nl = [ self.means, self.variances, self.mixing_coefficients ]
            c, convergence = convergence_function(pl,nl,c)
            pl=nl


def bayes_info_criterion(gmm):
    k = 3 * gmm.num_components
    n = np.size(gmm.image_matrix.flatten())
    L = gmm.likelihood()

    return np.log(n)*k - 2*L

def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    
    """
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563, 0.964706]
    ]

    kMin=2
    kMax=7
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)

    minBIC=0
    maxLic=0
    minBicModel = None
    maxLicModel = None
    for k in range(kMin,kMax+1):
        kk = k-kMin
        initial_means = comp_means[kk]
        num_components = k
        gmm = GaussianMixtureModel(image_matrix, num_components)
        gmm.initialize_training()
        gmm.means = np.array(initial_means)
        initial_liklihood = gmm.likelihood()
        initial_bic = bayes_info_criterion(gmm)
        gmm.train_model()

        l = gmm.likelihood()
        b = bayes_info_criterion(gmm)

        if(k==kMin):
            minBIC=b
            maxLic=l
            minBicModel=gmm
            maxLicModel=gmm
        else:
            if b<minBIC: 
                minBIC=b
                minBicModel=gmm
            if l>maxLic: 
                maxLic=l
                maxLicModel=gmm

    print("MinBIC: %d") % minBicModel.num_components
    print("MaxLic: %d") % maxLicModel.num_components

    return minBicModel, maxLicModel

def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    bic = 7
    likelihood = 7
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs
