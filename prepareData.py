import numpy as np
from scipy.io import loadmat
import os

def prepareOneSubj(fileName):
    dirPath = "DROZY/drozy/"
    data = loadmat(dirPath+fileName)
    y = data['labels'].ravel()
    return data['samples'], y


def prepareForML(leaveFileName):
    dirPath = "DROZY/drozy/"
    features = []
    labels = []
    domain_labels = np.array([])
    files = os.listdir(dirPath)
    jj = 0
    for i, fileName in enumerate(files):
        if(fileName==leaveFileName):
            continue
        # print(fileName)
        features_, labels_ = prepareOneSubj(fileName)
        if(len(features)==0):
            features = features_
            labels = labels_
        else:
            features = np.concatenate((features,features_))
            labels = np.concatenate((labels,labels_))
        domain_labels = np.concatenate((domain_labels,np.array([jj]*labels_.shape[0])))
        jj = jj + 1
    y = labels.ravel()
    return features, y, domain_labels