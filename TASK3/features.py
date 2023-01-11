# TUWIEN - WS2022 CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
import sklearn.metrics.pairwise as sklearn_pairwise
import cv2
import numpy as np
import random
import time


def extract_dsift(images: List[np.ndarray], stepsize: int, num_samples: int = None) -> List[np.ndarray]:
    """
    Extract dense feature points on a regular grid with 'stepsize' and if given, return
    'num_samples' random samples per image. if 'num_samples' is not given, take all features
    extracted with the given 'stepsize'. sift.compute has the argument "keypoints", set it to
    a list of keypoints for each square.
    HINT: cv2.Keypoint(...), cv2.SIFT_create(), sift.compute(img, keypoints), random.sample(..)
    images : list of images to extract dsift [num_of_images x n x m] - float
    stepsize : grid spacing, step size in x and y direction - int
    num_samples : random number of samples per image (optional) - int
    """


    tic = time.perf_counter()

    # student_code start
    # raise NotImplementedError("TO DO in features.py")
    all_descriptors = []

    # Build grid of stepsize spaced keypoints and take subsample
    width, height = images[0].shape
    grid = [cv2.KeyPoint(x+(stepsize/2), y+(stepsize/2), stepsize) for x in range(0, width, stepsize) for y in range(0, height, stepsize)]
    if num_samples != None:
        grid = random.sample(grid, num_samples)

    # Create cv2-sift object
    num_sift_features = 100
    sift = cv2.SIFT_create(num_sift_features)

    # For every image, get descriptors for keypoints and save them to all_descriptors
    for image in images:
        image_int = cv2.normalize(image, None, 0, 255.5, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, image_descriptors = sift.compute(image_int, grid)
        all_descriptors.append(image_descriptors)
    # student_code end

    toc = time.perf_counter()
    print("DSIFT Extraction:",  {toc - tic}, " seconds")

    # all_descriptors : list sift descriptors per image [number_of_images x num_samples x 128] - float
    return all_descriptors


def count_visual_words(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generate a histogram of word occurence per image
    Use sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
    to the nearest centroids and count the occurences of each centroids. The histogram
    should be as long as the vocabulary size (number of centroids)
    HINT: sklearn_pairwise.pairwise_distances(..), np.histogram(..)
    dense_feat : list sift descriptors per image [number_of_images x num_samples x 128] - float
    centroids : centroids of clusters [vocabulary_size x 128]
    """

    tic = time.perf_counter()

    # student_code start
    # raise NotImplementedError("TO DO in features.py")
    
    histograms = [] 

    # For every image look at computed features
    for image_features in dense_feat:
        '''
            pairwise_distances returns:
            Dndarray of shape (n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y)
                (...)
                If Y is not None, then D_{i, j} is the distance between the ith array from X and the jth array from Y.  
        '''
        image_histogram = np.zeros((len(centroids)), dtype=np.int64)
        distance_matrix = sklearn_pairwise.pairwise_distances(image_features, centroids)

        for i in range(len(image_features)):
            centroid_idx = np.argmin(distance_matrix[i])
            image_histogram[centroid_idx] += 1

        histograms.append(image_histogram)

    # student_code end

    toc = time.perf_counter()
    print("Counting visual words:",  {toc - tic}, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return histograms
