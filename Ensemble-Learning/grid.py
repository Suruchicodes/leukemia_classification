from __future__ import print_function
import os
import pandas as pd
import numpy as np
from sklearn.metrics import *
import math
from sklearn.preprocessing import label_binarize

#ROC-AUC
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))

def fuzzy_rank(CF, top):
    R_L = np.zeros(CF.shape)
    G_zero = 1 - alpha_1*math.exp(-alpha_2*math.exp(-alpha_3*0))
    #print('G_zero:',G_zero)
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            for k in range(CF.shape[2]):
                R_L[i][j][k] = 1 - alpha_1*math.exp(-alpha_2*math.exp(-alpha_3*CF[i][j][k]))  #Gompertz Function
                #print(f'Rank[{i}][{j}][{k}]:', R_L[i][j][k])

    K_L = G_zero*np.ones(shape = R_L.shape) #initiate all values as penalty values

    for i in range(R_L.shape[0]):
        for sample in range(R_L.shape[1]):
            for k in range(top):
                a = R_L[i][sample]
                idx = np.where(a==np.partition(a, k)[k])
                #if sample belongs to top 'k' classes, R_L =R_L, else R_L = penalty value
                K_L[i][sample][idx] = R_L[i][sample][idx]

    return K_L

def CFS_func(CF, K_L):
    H = CF.shape[0] #no. of classifiers
    G_zero = 1 - alpha_1*math.exp(-alpha_2*math.exp(-alpha_3*0))
    for f in range(CF.shape[0]):
        for i in range(CF.shape[1]):
            idx = np.where(K_L[f][i] == G_zero)
            CF[f][i][idx] = 0
    CFS = 1 - np.sum(CF,axis=0)/H
    #print('CFS:', CFS)
    return CFS

def Gompertz(top, alpha_1, alpha_2, alpha_3, *argv):
    L = 0 #Number of classifiers
    G_initial = 1 - alpha_1*math.exp(-alpha_2*math.exp(-alpha_3*0))
    for arg in argv:
        L += 1

    num_classes = arg.shape[1]
    CF = np.zeros(shape = (L,arg.shape[0], arg.shape[1]))

    for i, arg in enumerate(argv):
        CF[:][:][i] = arg

    R_L = fuzzy_rank(CF, top) #R_L is with penalties

    RS = np.sum(R_L, axis=0)
    #print('RS', RS)
    CFS = CFS_func(CF, R_L)
    #print('CFS', CFS)
    FS = RS*CFS
    #print('FS', FS)
    predictions = np.argmin(FS,axis=1)
    return predictions


# Load the predictions and ground truth labels
p1 = pd.read_csv('Model_1_pred.csv', header=None)
p2 = pd.read_csv('Model_2_pred.csv', header=None)
p3 = pd.read_csv('Model_3_pred.csv', header=None)
p4 = pd.read_csv('Model_4_pred.csv', header=None)
p5 = pd.read_csv('Model_5_pred.csv', header=None)
labels = pd.read_csv('GT_labels.csv', header=None)

# Remove unnecessary rows and columns
p1 = np.asarray(p1)[1:, 1:]
p2 = np.asarray(p2)[1:, 1:]
p3 = np.asarray(p3)[1:, 1:]
p4 = np.asarray(p4)[1:, 1:]
p5 = np.asarray(p5)[1:, 1:]
labels = np.asarray(labels)[1:, 1:]

pred = [tuple(np.argmax(p, axis=1)) for p in [p1, p2, p3, p4, p5]]


classes = []
for i in range(p1.shape[1]):
    classes.append(str(i+1))
acc =[]
for top in range(1, 3):  # Reduced range for 'top'
    for alpha_1 in range(1, 6, 2):  # Reduced range and step size for 'alpha_1'
        for alpha_2 in range(1, 6, 2):  # Reduced range and step size for 'alpha_2'
            for alpha_3 in range(1, 6, 2):  # Reduced range and step size for 'alpha_3'
                predictions = Gompertz(top, alpha_1, alpha_2, alpha_3, p1, p2, p3, p4, p5)
                predictions = np.squeeze(predictions)
                labels = np.squeeze(labels)
                correct = np.where(predictions == labels)[0].shape[0]
                total = labels.shape[0]

                accuracy = correct / total
                print(f"top = {top}, Alpha_1 = {alpha_1}, Alpha_2 = {alpha_2}, Alpha_3 = {alpha_3}")
                if accuracy >= 0.888:
                    print("Accuracy = ", correct / total)
                acc.append(accuracy)

print("Max Accuracy = ", max(acc))