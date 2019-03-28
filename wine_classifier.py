#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from utilities import load_data, print_features, print_predictions
from sklearn.decomposition import PCA

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    

###
# New Functions
###
def plot_pairwise(train_labels, train_set, data_size, num_features, class_colours, colours_size):
    n_features = train_set.shape[1]
    fig, ax = plt.subplots(n_features, n_features, figsize=(30, 20))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.5, hspace=0.5)
    color_mat = []
    for i in range (0, data_size):
        if train_labels[i] == 1:
            color_mat.append(CLASS_1_C)
        elif train_labels[i] == 2:
            color_mat.append(CLASS_2_C)
        else:
            color_mat.append(CLASS_3_C)
    for x in range(0,num_features):
        for y in range(0,num_features):
            ax[x][y].scatter(train_set[:, x], train_set[:, y], c=color_mat, s=1)
            ax[x][y].set_title('Features ' + str (x+1) + ' vs '+ str (y+1))
    plt.show()

def nearest_neighbors(train_set, test_set, test_labels, k):
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
    train_dist = lambda x : [dist(x, point) for point in train_set]
    predicted = [train_dist(p) for p in test_set]
    predicted = np.argsort(predicted)
    results = []
    
    for i in range(0,53):
        if k > 1:
            temp = []
            for j in range(0,k):
                temp.append(test_labels[predicted[i][j]].astype(np.int))
            temp = stats.mode(temp, axis=None)
            results.append(temp[0][0])
        else:
            results.append(test_labels[predicted[i][0]].astype(np.int))
    
    return results

def calculate_confusion_matrix(gt_labels, pred_labels):
    # write your code here (remember to return the confusion matrix at the end!)  
    # Create nxn matrix
    confusion_matrix = np.zeros((np.unique(gt_labels).size,np.unique(gt_labels).size))
    
    counts = np.zeros(np.unique(gt_labels).size)
    
    for i in range(0, len(pred_labels)):
        confusion_matrix[(gt_labels[i] - 1), (pred_labels[i] - 1)] += 1
        counts[(gt_labels[i] - 1)] += 1

    for i in range(0, np.unique(gt_labels).size):
         confusion_matrix[i] /= counts[i]

    for i in range (0, 3):
        for j in range (0, 3):
            confusion_matrix[i][j] = round(confusion_matrix[i][j], 2)

    return confusion_matrix

def plot_matrix(matrix, ax=None):  
    if ax is None:
        ax = plt.gca()
        
    # write your code here
    plt.imshow(matrix,cmap=plt.get_cmap('summer'))
    plt.colorbar()
    for i in range(0,np.size(matrix,0)):
        for j in range(0,np.size(matrix,1)):
            plt.text(i,j,matrix[j,i], va = "center", ha = "center")
    plt.title('Confusion matrix')
    plt.show()
###
# Skeleton Code
###
def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of 
    # the function
    class_colours = [CLASS_1_C,CLASS_2_C,CLASS_3_C]
    plot_pairwise(train_labels, train_set, 125, 13, class_colours,3)
    return [7,10]

def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    train_set_selected = np.column_stack((train_set[:,6], train_set[:,9]))
    test_set_selected = np.column_stack((test_set[:,6], test_set[:,9]))                                      
    
    result_labels = nearest_neighbors(train_set_selected, test_set_selected, train_labels, k)
    
    return result_labels


def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    train_set_selected = np.column_stack((train_set[:,6], train_set[:,9], train_set[:,11]))
    test_set_selected = np.column_stack((test_set[:,6], test_set[:,9], test_set[:,11]))                                      
    
    return nearest_neighbors(train_set_selected, test_set_selected, train_labels, k)


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    pca = PCA(n_components=n_components)
    pca.fit(train_set)
    pca_transformed_training_set = pca.transform(train_set)
    pca_transformed_test_set = pca.transform(test_set)

    neg_ones = np.full(len(pca_transformed_training_set), -1)
    pca_transformed_training_set_xs = pca_transformed_training_set[:, 0]
    pca_transformed_training_set_ys = pca_transformed_training_set[:, 1]
    neg_pca_transformed_training_set_ys = np.multiply(pca_transformed_training_set_ys, neg_ones)

    color_mat = []
    for i in range (0, 125):
        if train_labels[i] == 1:
            color_mat.append(CLASS_1_C)
        elif train_labels[i] == 2:
            color_mat.append(CLASS_2_C)
        else:
            color_mat.append(CLASS_3_C)

    train_set_selected = np.column_stack((train_set[:, 6], train_set[:, 9]))
    test_set_selected = np.column_stack((test_set[:, 6], test_set[:, 9]))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(train_set_selected[:, 0], train_set_selected[:, 1], c=color_mat)
    ax[0].set_title('Reduced selecting features (7, 10)')
    ax[1].scatter(pca_transformed_training_set_xs, neg_pca_transformed_training_set_ys, c=color_mat)
    ax[1].set_title("Reduced with Scipy's PCA")

    plt.show()

    return nearest_neighbors(pca_transformed_training_set, pca_transformed_test_set, train_labels, k)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    #Added by Cameron
    elif mode == 'knn_confusion':
        result_labels = knn(train_set, train_labels, test_set, args.k)
        confusion_matrix = calculate_confusion_matrix(test_labels, result_labels)
        print(confusion_matrix)
        plot_matrix(confusion_matrix)
    elif mode == 'knn_3d_confusion':
        result_labels = knn_three_features(train_set, train_labels, test_set, args.k)
        confusion_matrix = calculate_confusion_matrix(test_labels, result_labels)
        print(confusion_matrix)
        plot_matrix(confusion_matrix)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))