from skimage import io, color
import superpixel as sp
import scipy.io, sys
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
from featureExtract import Feature
import glob
#import maxflow
from skimage.util import img_as_float
import argparse
import numpy as np
from pystruct.learners import NSlackSSVM
from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger
from time import time
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('train_db_path', help='Path to training database')
parser.add_argument('test_path', help='Path to test database')
arguments = parser.parse_args()

data = scipy.io.loadmat(arguments.train_db_path)
test = scipy.io.loadmat(arguments.test_path)
train_data = data['train_data']
valid_data = data['valid_data']
validationOriginalImage = data['validationOriginalImage']
valid_superpixels = data['valid_superpixels']
valid_edgesFeatures1 = data['valid_edgesFeatures1']
valid_edgesFeatures2 = data['valid_edgesFeatures2']
test_edgesFeatures1 = test['test_edgesFeatures1']
test_edgesFeatures2 = test['test_edgesFeatures2']
valid_edges = data['valid_edges']
train_labels = data['train_labels']
valid_labels = data['valid_labels']
test_data = test['test_data']
test_labels = test['test_label']
test_edges = test['test_edges']
# Preprocessing normalize data
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
#Preprocessing RandomizePCA
#pca = RandomizedPCA(n_components=15)
#pca.fit(train_data)
scaler.fit(valid_data)
valid_data = scaler.transform(valid_data)
scaler.fit(test_data)
test_data = scaler.transform(test_data)
#valid_data = pca.transform(valid_data)
clf = knn(n_neighbors=21, p=1)
clf = clf.fit(train_data,train_labels.ravel())
print clf.score(valid_data,valid_labels.ravel())
print clf.score(test_data,test_labels.ravel())
"""
for file_num in range(210,213):#test_files_count):
    # see test results
    sp_file_names = data['sp_file_names'][file_num].strip()
    im_file_names = data['im_file_names'][file_num].strip()

    # Extract features from image files
    fe = Feature()
    fe.loadImage(im_file_names)
    fe.loadSuperpixelImage()
    test_data = fe.getFeaturesVectors()
   # edges, feat = fe.getEdges()
    # Normalize data
    test_data = scaler.transform(test_data)
    #test_data = pca.transform(test_data)

    sp.showPrediction(file_num, clf, fe.getSuperpixelImage(), test_data, fe.getImage())
    """
valid_count = 0
#for i in range(0,len(valid_edges)):
#    print valid_edges[i][0].shape
x_valid = []
for i in range(0, valid_data.shape[0]):
    temp = np.zeros((1,2),dtype = int)
#    print clf.predict_proba(valid_data[i])[0]
    x_valid.append(clf.predict_proba(valid_data[i])[0])
unary_file = []
for i in range(0, len(valid_edges)):
    unary = []
    for j in range (0, valid_edges[i][0].shape[0]):
        unary.append(x_valid[valid_count + j])
    valid_count = valid_count + valid_edges[i][0].shape[0]
    unary = np.array(unary)
    unary_file.append(unary)
X_valid = []
for i in range(0,len(unary_file)):
    edges = []
    edgesFeatures = []
    for j in range(0, (valid_edges[i][0]).shape[0]):
        for k in range(j,(valid_edges[i][0]).shape[1]):
           if  (valid_edges[i][0])[j][k] == 1:
               edges.append([j,k])
               edgesFeatures.append([1.0, valid_edgesFeatures1[i][0][j][k],valid_edgesFeatures2[i][0][j][k]])
    edges = np.array(edges)
    edgesFeatures = np.array(edgesFeatures)
    X_valid.append((np.atleast_2d(unary_file[i]), np.array(edges, dtype=np.int),edgesFeatures))
print len(X_valid)
print len(X_valid[0])
print type(X_valid[0][0])
print X_valid[0][0].shape
print type(X_valid[0][1])
print X_valid[0][1].shape
valid_Y = []
valid_count = 0
for i in range (0, len(valid_edges)):
    labels = np.zeros([1,valid_edges[i][0].shape[0]],dtype = int)
    for j in range (0, valid_edges[i][0].shape[0]):
        labels[0][j] = valid_labels[valid_count + j].astype(int)
    valid_count = valid_count + valid_edges[i][0].shape[0]
    valid_Y.append(labels[0])
x_test = []
test_count = 0
for i in range(0, test_data.shape[0]):
    x_test.append(clf.predict_proba(test_data[i])[0])
unary_file = []
for i in range(0, len(test_edges)):
    unary = []
    for j in range (0, test_edges[i][0].shape[0]):
        unary.append(x_test[(test_count + j)])
    test_count = test_count +test_edges[i][0].shape[0]
    unary = np.array(unary)
    unary_file.append(unary)
X_test = []
for i in range(0,len(unary_file)):
    edges = []
    edgesFeatures = []
    for j in range(0, test_edges[i][0].shape[0]):
        for k in range (j,test_edges[i][0].shape[1]):
           if  (test_edges[i][0])[j][k] == 1:
               edges.append([j,k])
               edgesFeatures.append([1.0, test_edgesFeatures1[i][0][j][k],test_edgesFeatures2[i][0][j][k]])
    edges = np.array(edges)
    edgesFeatures = np.array(edgesFeatures)
    X_test.append((np.atleast_2d(unary_file[i]), np.array(edges, dtype=np.int),edgesFeatures))
test_count = 0
test_Y = []
for i in range (0, len(test_edges)):
    labels = np.zeros([1,test_edges[i][0].shape[0]],dtype = int)
    for j in range (0, test_edges[i][0].shape[0]):
        labels[0][j] = test_labels[test_count + j].astype(int)
    test_count = test_count + test_edges[i][0].shape[0]
    test_Y.append(labels[0])
C = 0.01
n_states = 2
class_weights = 1. / np.bincount(np.hstack(valid_Y))
class_weights *= 2. / np.sum(class_weights)
print(class_weights)

model = crfs.EdgeFeatureGraphCRF(class_weight=class_weights)

experiment_name = "edge_features_one_slack_trainval_%f" % C

ssvm = learners.NSlackSSVM(
    model, verbose=2, C=1, max_iter=1000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    inactive_threshold=1e-3, inactive_window=10, batch_size=100)
ssvm.fit(X_valid, valid_Y)

print ssvm.score(X_valid,valid_Y)
print ssvm.score(X_test,test_Y)
predict = ssvm.predict(X_valid)
"""
for i in range(0, len(X_valid)):
    predict_result = predict[i]
    fe = Feature()
    x = glob.glob("../data_road/training/image/um_000001.png")
    print len(x)
    fe.loadImage(x[0])
    fe.loadSuperpixelImage()
    image = fe.getImage()
    superpixels = valid_superpixels[i][0]
    newIm = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        indices = np.where(superpixels==i)
        prediction = predict_result[i]
        image[indices] = 1
    #sp.showPlots("im_name", image, numSuperpixels, superpixels)
#superpixels = 
#sp.showPlots(x, y_pred[0], np.max(superpixels),superpixels):
#print y_pred

# we throw away void superpixels and flatten everything
y_pred, y_true = np.hstack(y_pred), np.hstack(test_Y)
y_pred = y_pred[y_true != 255]
y_true = y_true[y_true != 255]

print("Score on validation set: %f" % np.mean(y_true == y_pred))
"""