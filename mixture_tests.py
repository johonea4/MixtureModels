from helper_functions import image_to_matrix, matrix_to_image, flatten_image_matrix
import numpy as np
from mixture_models import k_means_cluster, image_difference, GaussianMixtureModel, bonus, bayes_info_criterion, \
    GaussianMixtureModelConvergence, GaussianMixtureModelImproved


def k_means_test():
    """
    Testing your implementation
    of k-means on the segmented
    bird_color_24 reference images.
    """
    k_min = 2
    k_max = 6
    image_dir = 'images/'
    image_name = 'bird_color_24.png'
    image_values = image_to_matrix(image_dir + image_name)
    # initial mean for each k value
    initial_means = [
        np.array([[0.90980393, 0.8392157, 0.65098041], [0.83137256, 0.80784315, 0.69411767]]),
        np.array([[0.90980393, 0.8392157, 0.65098041], [0.83137256, 0.80784315, 0.69411767],
                  [0.67450982, 0.52941179, 0.25490198]]),
        np.array([[0.90980393, 0.8392157, 0.65098041], [0.83137256, 0.80784315, 0.69411767],
                  [0.67450982, 0.52941179, 0.25490198], [0.86666667, 0.8392157, 0.70588237]]),
        np.array([[0.90980393, 0.8392157, 0.65098041], [0.83137256, 0.80784315, 0.69411767],
                  [0.67450982, 0.52941179, 0.25490198], [0.86666667, 0.8392157, 0.70588237], [0, 0, 0]]),
        np.array([[0.90980393, 0.8392157, 0.65098041], [0.83137256, 0.80784315, 0.69411767],
                  [0.67450982, 0.52941179, 0.25490198], [0.86666667, 0.8392157, 0.70588237], [0, 0, 0],
                  [0.8392157, 0.80392158, 0.63921571]]),
    ]
    # test different k values to find best
    for k in range(k_min, k_max + 1):
        updated_values = k_means_cluster(image_values, k, initial_means[k - k_min])
        ref_image = image_dir + 'k%d_%s' % (k, image_name)
        ref_values = image_to_matrix(ref_image)
        dist = image_difference(updated_values, ref_values)
        print('Image distance = %.2f' % dist)
        if int(dist) == 0:
            print('Clustering for %d clusters produced a realistic image segmentation.' % k)
        else:
            print('Clustering for %d clusters didn\'t produce a realistic image segmentation.' % k)


def gmm_likelihood_test():
    """Testing the GMM method
    for calculating the overall
    model probability.
    Should return -364370.

    returns:
    likelihood = float
    """
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    num_components = 5
    gmm = GaussianMixtureModel(image_matrix, num_components)
    gmm.initialize_training()
    gmm.means = [0.4627451, 0.10196079, 0.027450981, 0.011764706, 0.1254902]
    likelihood = gmm.likelihood()
    return likelihood


def gmm_joint_prob_test():
    """Testing the GMM method
    for calculating the joint
    log probability of a given point.
    Should return -0.98196.

    returns:
    joint_prob = float
    """
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    num_components = 5
    gmm = GaussianMixtureModel(image_matrix, num_components)
    gmm.initialize_training()
    gmm.means = [0.4627451, 0.10196079, 0.027450981, 0.011764706, 0.1254902]
    test_val = 0.4627451
    joint_prob = gmm.joint_prob(0.4627451)
    return joint_prob


def generate_test_mixture(data_size, means, variances, mixing_coefficients):
    """
    Generate synthetic test
    data for a GMM based on
    fixed means, variances and
    mixing coefficients.

    params:
    data_size = (int)
    means = [float]
    variances = [float]
    mixing_coefficients = [float]

    returns:
    data = np.array[float]
    """

    data = np.zeros(data_size).flatten()

    indices = np.random.choice(len(means), len(data), p=mixing_coefficients)

    for i in range(len(indices)):
        data[i] = np.random.normal(means[indices[i]], variances[indices[i]])

    return np.array([data])


def gmm_train_test():
    """Test the training
    procedure for GMM using
    synthetic data.

    returns:
    gmm = GaussianMixtureModel
    """

    print('Synthetic example with 2 means')

    num_components = 2
    data_range = (1, 1000)
    actual_means = [2, 4]
    actual_variances = [1] * num_components
    actual_mixing = [.5] * num_components
    dataset_1 = generate_test_mixture(data_range, actual_means, actual_variances, actual_mixing)
    gmm = GaussianMixtureModel(dataset_1, num_components)
    gmm.initialize_training()
    # start off with faulty means
    gmm.means = [1, 3]
    initial_likelihood = gmm.likelihood()

    gmm.train_model()
    final_likelihood = gmm.likelihood()
    likelihood_difference = final_likelihood - initial_likelihood
    likelihood_thresh = 250
    if likelihood_difference >= likelihood_thresh:
        print('Congrats! Your model\'s log likelihood improved by at least %d.' % likelihood_thresh)

    print('Synthetic example with 4 means:')

    num_components = 4
    actual_means = [2, 4, 6, 8]
    actual_variances = [1] * num_components
    actual_mixing = [.25] * num_components
    dataset_1 = generate_test_mixture(data_range,
                                      actual_means, actual_variances, actual_mixing)
    gmm = GaussianMixtureModel(dataset_1, num_components)
    gmm.initialize_training()
    # start off with faulty means
    gmm.means = [1, 3, 5, 9]
    initial_likelihood = gmm.likelihood()
    gmm.train_model()
    final_likelihood = gmm.likelihood()

    # compare likelihoods
    likelihood_difference = final_likelihood - initial_likelihood
    likelihood_thresh = 200
    if likelihood_difference >= likelihood_thresh:
        print('Congrats! Your model\'s log likelihood improved by at least %d.' % likelihood_thresh)
    return gmm


def gmm_segment_test():
    """
    Apply the trained GMM
    to unsegmented image and
    generate a segmented image.

    returns:
    segmented_matrix = numpy.ndarray[numpy.ndarray[float]]
    """
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    num_components = 3
    gmm = GaussianMixtureModel(image_matrix, num_components)
    gmm.initialize_training()
    gmm.train_model()
    segment = gmm.segment()
    segment_num_components = len(np.unique(segment))
    if segment_num_components == num_components:
        print('Congrats! Your segmentation produced an image ' +
              'with the correct number of components.')
    return segment


def gmm_best_segment_test():
    """
    Calculate the best segment
    generated by the GMM and
    compare the subsequent likelihood
    of a reference segmentation.
    Note: this test will take a while
    to run.

    returns:
    best_seg = np.ndarray[np.ndarray[float]]
    """
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    image_matrix_flat = flatten_image_matrix(image_matrix)
    num_components = 3
    gmm = GaussianMixtureModel(image_matrix, num_components)
    gmm.initialize_training()
    iters = 10
    # generate best segment from 10 iterations
    # and extract its likelihood
    best_seg = gmm.best_segment(iters)
    matrix_to_image(best_seg, 'images/best_segment_spock.png')
    best_likelihood = gmm.likelihood()

    # extract likelihood from reference image
    ref_image_file = 'images/party_spock%d_baseline.png' % num_components
    ref_image = image_to_matrix(ref_image_file, grays=True)
    gmm_ref = GaussianMixtureModel(ref_image, num_components)
    ref_vals = ref_image.flatten()
    ref_means = list(set(ref_vals))
    ref_variances = [0] * num_components
    ref_mixing = [0] * num_components
    for i in range(num_components):
        relevant_vals = ref_vals[ref_vals == ref_means[i]]
        ref_mixing[i] = float(len(relevant_vals)) / float(len(ref_vals))
        ref_variances[i] = np.mean((image_matrix_flat[ref_vals == ref_means[i]] - ref_means[i]) ** 2)
    gmm_ref.means = ref_means
    gmm_ref.variances = ref_variances
    gmm_ref.mixing_coefficients = ref_mixing
    ref_likelihood = gmm_ref.likelihood()

    # compare best likelihood and reference likelihood
    likelihood_diff = best_likelihood - ref_likelihood
    likelihood_thresh = 1e4
    if likelihood_diff >= likelihood_thresh:
        print('Congrats! Your image segmentation is an improvement over ' +
              'the baseline by at least %.2f.' % likelihood_thresh)
    return best_seg


def gmm_improvement_test():
    """
    Tests whether the new mixture
    model is actually an improvement
    over the previous one: if the
    new model has a higher likelihood
    than the previous model for the
    provided initial means.

    returns:
    original_segment = numpy.ndarray[numpy.ndarray[float]]
    improved_segment = numpy.ndarray[numpy.ndarray[float]]
    """
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    num_components = 3
    initial_means = [0.4627451, 0.20392157, 0.36078432]
    # first train original model with fixed means
    gmm = GaussianMixtureModel(image_matrix, num_components)
    gmm.initialize_training()
    gmm.means = np.copy(initial_means)
    gmm.train_model()
    original_segment = gmm.segment()
    original_likelihood = gmm.likelihood()
    # then train improved model
    gmm_improved = GaussianMixtureModelImproved(image_matrix, num_components)
    gmm_improved.initialize_training()
    gmm_improved.train_model()
    improved_segment = gmm_improved.segment()
    improved_likelihood = gmm_improved.likelihood()
    # then calculate likelihood difference
    diff_thresh = 1e3
    likelihood_diff = improved_likelihood - original_likelihood
    if likelihood_diff >= diff_thresh:
        print('Congrats! Improved model scores a likelihood that was at ' +
              'least %d higher than the original model.' % diff_thresh)
    return original_segment, improved_segment


def convergence_condition_test():
    """
    Compare the performance of
    the default convergence function
    with the new convergence function.

    return:
    default_convergence_likelihood = float
    new_convergence_likelihood = float
    """
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    num_components = 3
    initial_means = [0.4627451, 0.10196079, 0.027450981]

    # first test original model
    gmm = GaussianMixtureModel(image_matrix, num_components)
    gmm.initialize_training()
    gmm.means = np.copy(initial_means)
    gmm.train_model()
    default_convergence_likelihood = gmm.likelihood()

    # now test new convergence model
    gmm_new = GaussianMixtureModelConvergence(image_matrix, num_components)
    gmm_new.initialize_training()
    gmm_new.means = np.copy(initial_means)
    gmm_new.train_model()
    new_convergence_likelihood = gmm_new.likelihood()

    # test convergence difference
    convergence_diff = new_convergence_likelihood - default_convergence_likelihood
    convergence_thresh = 200
    if convergence_diff >= convergence_thresh:
        print('Congrats! The likelihood difference between the original '
              + 'and the new convergence models should be at least %.2f' % convergence_thresh)
    return default_convergence_likelihood, new_convergence_likelihood


def bayes_info_test():
    """
    Test for your
    implementation of
    BIC on fixed GMM values.
    Should be about 727045.

    returns:
    BIC = float
    """

    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    num_components = 3
    initial_means = [0.4627451, 0.10196079, 0.027450981]
    gmm = GaussianMixtureModel(image_matrix, num_components)
    gmm.initialize_training()
    gmm.means = np.copy(initial_means)
    b_i_c = bayes_info_criterion(gmm)
    return b_i_c


def bonus_test():
    points = np.array(
        [[0.9059608, 0.67550357, 0.13525533], [0.23656114, 0.63624466, 0.3606615], [0.91163215, 0.24431103, 0.33318504],
         [0.25209736, 0.24600123, 0.42392935], [0.62799146, 0.04520208, 0.55232494],
         [0.5588561, 0.06397713, 0.53465371], [0.82530045, 0.62811624, 0.79672349],
         [0.50048147, 0.13215356, 0.54517893], [0.84725662, 0.71085917, 0.61111105],
         [0.25236734, 0.25951904, 0.70239158]])
    means = np.array([[0.39874413, 0.47440682, 0.86140829], [0.05671347, 0.26599323, 0.33577454],
                      [0.7969679, 0.44920099, 0.37978416], [0.45428452, 0.51414022, 0.21209852],
                      [0.7112214, 0.94906158, 0.25496493]])
    expected_answer = np.array([[0.90829883, 0.9639127, 0.35055193, 0.48575144, 0.35649377],
                                [0.55067427, 0.41237201, 0.59110637, 0.29048911, 0.57821151],
                                [0.77137409, 0.8551975, 0.23937264, 0.54464354, 0.73685561],
                                [0.51484192, 0.21528078, 0.58320052, 0.39705222, 0.85652654],
                                [0.57645778, 0.64961631, 0.47067874, 0.60483973, 0.95515036],
                                [0.54850426, 0.57663736, 0.47862222, 0.56358129, 0.94064631],
                                [0.45799673, 0.966609, 0.45458971, 0.70173336, 0.63993928],
                                [0.47695785, 0.50861901, 0.46451987, 0.50891112, 0.89217387],
                                [0.56543953, 0.94798437, 0.35285421, 0.59357932, 0.4495398],
                                [0.30477736, 0.41560848, 0.66079087, 0.58820896, 0.94138546]])
    if np.allclose(expected_answer, bonus(points, means), 1e-7):
        print 'You returned the correct distances.'
    else:
        print 'Your distance calculation is incorrect.'

if __name__ == '__main__':
    k_means_test()
    # gmm_likelihood_test()
    # gmm_joint_prob_test()
    # gmm_train_test()
    # gmm_segment_test()
    # gmm_best_segment_test()
    # gmm_improvement_test()
    # convergence_condition_test()
    # bayes_info_test()
    # bonus_test()