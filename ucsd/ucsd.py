import sys
sys.path.insert(0, '../lib')
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

#
# video_path = 'video-1567808222.mp4'
# # video_path = '../../resources/UCSD-Nat/8.mp4'
#
# video = cv2.VideoCapture(video_path)
#
# fc = video.get(cv2.CAP_PROP_FRAME_COUNT)
#
# buffer = []
# count = 0
# while video.isOpened():
#     count += 1
#     ret_val, frame = video.read()
#     print(count)
#     if (not ret_val) or (cv2.waitKey(1) == 27) or count > 280:
#         break
#     if count > 20:
#         buffer.append(frame)
#
# import median
#
# m = median.Median()
# mi = m.get_median(buffer)
#
# # mi = cv2.cvtColor(mi, cv2.COLOR_BGR2RGB)
# # plt.imshow(mi)
# # plt.show()
#
#
# timebuffer = np.float32(buffer[20:250]).reshape(len(buffer[20:250]), 352*640, 3)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# timeclusters = cv2.kmeans(timebuffer, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[2]
# timeclusters = np.array([np.uint8(timecenter.reshape(352, 640, 3)) for timecenter in timeclusters])
#
#
# video.open(video_path)
# video.set(cv2.CAP_PROP_POS_FRAMES, 20)
#
# import time
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('./out/detectedImage'+str(int(time.time()))+'.avi', fourcc, video.get(cv2.CAP_PROP_FPS), (640, 352))
#
# from skimage import filters
# import blobParams
# b = blobParams.BlobParams()
# import centroidTracker
# ct = centroidTracker.CentroidTracker()
# frame_count = 0
# while video.isOpened():
#     frame_count += 1
#     print(frame_count)
#     ret_val, frame = video.read()
#     if (not ret_val) or (cv2.waitKey(1)==27):
#         break
#     d = [cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY) for bg in timeclusters]
#     d = np.array(d).min(axis=0)
#     cv2.imshow('d', d)
#     fgh = 255*np.uint8(filters.apply_hysteresis_threshold(d, 15, 50))
#     cv2.imshow('fgh', fgh)
#     fgt = 255*np.uint8(filters.threshold_local(d, 25, offset=20))
#     cv2.imshow('fgt', fgt)
#     detected_blobs = b.get_blob(fgh, 185)
#     rects, output = b.get_bounding_rectangles(frame, detected_blobs)
#     objects = ct.update(rects)
#     b.get_ids(output, objects)
#     cv2.imshow('DetectedImage', output)
#     out.write(output)
# # cv2.destroyAllWindows()
#
#
# video.release()

import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import random
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

#Import Features CSV file using pandas
file_name = 'annData.txt'
data_frame = pd.read_csv(file_name, sep="   ", names = ["Index", "Speed", "Position",
                                                        "Submersion Index", "Submersion Variance",
                                                        "Splash", "Activity Index", "Output"])
print(data_frame.head())

data_frame = data_frame.dropna()

#Understand the Raw Dataset
print("\nNumber of rows (Total number of dataset) in given feature file: " + str(data_frame.shape[0]))
print("Number of columns (Total number of features) in given feature file: " + str(data_frame.shape[1]))

def show_norm_gau_of_feature(data_frame_feature):
    sns.distplot(data_frame_feature, fit = norm, kde = False)
    plt.show()

def show_univariate_distribution_flexble(data_frame_feature):
    sns.distplot(data_frame_feature)
    plt.show()

show_univariate_distribution_flexble(data_frame["Speed"])

corr = data_frame[["Index", "Speed", "Position", "Submersion Index", "Submersion Variance",
                   "Splash", "Activity Index", "Output"]].corr()

def show_heatmap(corr_from_file):
    sns.heatmap(corr_from_file)
    plt.show()

show_heatmap(corr)

#Feature preprocessing if needed

drop_column = ['Index', 'Output']
X_feature_space = data_frame.drop(drop_column, axis=1)


#Get ground_truth and apply one-hot coding
y_ground_truth = data_frame["Output"]

def standardization_test(train_matrix, test_matrix):
    x_train_matrix = np.array(train_matrix)
    x_test_matrix = np.array(test_matrix)
    print(x_train_matrix)
    x_mean = x_train_matrix.sum(axis = 0) / x_train_matrix.shape[0]
    print(x_mean)
    x_std = np.sqrt(((x_train_matrix-x_mean)**2).sum(axis=0) / x_train_matrix.shape[0])
    print(x_std)
    if not os.path.exists(Train_Mean_Path) or not os.path.exists(Train_Std_Path):
        np.save(Train_Mean_Path, x_mean)
        np.save(Train_Std_Path, x_std)
    train_stdzat = (x_train_matrix - x_mean) / x_std
    test_stdzat = (x_test_matrix - x_mean) / x_std
    return train_stdzat, test_stdzat

#Model Training and Result Evaluation
ANN_Model_Path = 'ANN_Model.npy'
X_Train_Path = 'X_Train.npy'
Y_Train_Path = 'Y_Train.npy'
X_Test_Path = 'X_Test.npy'
Y_Test_Path = 'Y_Test.npy'
X1_Test_Path = 'X1_Test.npy'
Train_Mean_Path = 'Train_Mean.npy'
Train_Std_Path = 'Train_Std.npy'
if not os.path.exists(ANN_Model_Path):
    x_train_ori, x_test, y_train, y_test = model_selection.train_test_split(X_feature_space, y_ground_truth)
    x_train, x_test = standardization_test(x_train_ori, x_test)
    x1 = np.array([[1.7223130e-01, 6.1931826e+01, 7.4554722e-02,
                   6.7288089e-01, 6.8041544e-02, 8.6643994e-01],
                  [3.8890055e-02, 2.1595719e+01, 7.7176080e-03,
                   7.5837305e-02, 1.0487424e-01, 4.2562320e-02],
                  [2.2666759e-01, 7.0483095e+01, 7.1463925e-02,
                   6.3008836e-02, 5.5187415e-02, 3.9298553e-02],
                  [1.0625489e-01, 5.5562917e-01, 9.3402372e-02,
                   6.3901003e-03, 1.0420951e-01, 2.1379018e-02],
                  [1.3226214e-01, 1.8500301e+01, 7.1754457e-02,
                   6.6085574e-02, 7.9599375e-02, 1.6758212e-02]]).reshape(5, 6)
    x_train, x1_test = standardization_test(x_train_ori, x1)
    np.save(X_Train_Path, x_train)
    np.save(Y_Train_Path, y_train)
    np.save(X_Test_Path, x_test)
    np.save(Y_Test_Path, y_test)
    np.save(X1_Test_Path, x1_test)
else:

    x_train = np.load(X_Train_Path)
    x_test = np.load(X_Test_Path)
    y_train = np.load(Y_Train_Path)
    y_test = np.load(Y_Test_Path)
    x1_test = np.load(X1_Test_Path)


print(y_train.shape)
print(y_test.shape)

print("\nTraining dataset has " + str(x_train.shape[0]) + " oberservation with "
      + str(x_train.shape[1]) + " features.")
print("Testing dataset has " + str(x_test.shape[0]) + " oberservation with "
      + str(x_test.shape[1]) + " features.")

# Every Dataset Should Use Standardization
# Function Output is train_standarization matrix and test_standarization matrix
def standardization(train_matrix):
    x_train_matrix = np.array(train_matrix)
    # x_test_matrix = np.array(test_matrix)
    x_mean = x_train_matrix.sum(axis = 0) / x_train_matrix.shape[0]
    x_std = np.sqrt(((x_train_matrix-x_mean)**2).sum(axis=0) / x_train_matrix.shape[0])
    train_stdzat = (x_train_matrix - x_mean) / x_std
    # test_stdzat = (x_test_matrix - x_mean) / x_std
    return train_stdzat


#Activation Function Softmax Activation Function and RELU Function is Used here
def softmax(y_i_ascol):
    Max = y_i_ascol.max(axis = 0)
    return np.exp(y_i_ascol - Max) / (np.exp(y_i_ascol - Max)).sum(axis = 0)
def softmax_backp(a, error):
    act_g = softmax(a)
    inner_product = (act_g * error).sum(axis = 0)
    return act_g * error - inner_product * act_g
def relu(a):
    return a * (a > 0)
def relu_backp(a, e):
    return 1 * (a > 0) * e

def init_shallow(Ni, Nh, No):
    b1 = np.random.randn(Nh, 1) / np.sqrt((Ni+1.)/2.)
    W1 = np.random.randn(Nh, Ni) / np.sqrt((Ni+1.)/2.)
    b2 = np.random.randn(No, 1) / np.sqrt((Nh+1.))
    W2 = np.random.randn(No, Nh) / np.sqrt((Nh+1.))
    return W1, b1, W2, b2

def forwardprop_shallow(xtrain, net):
    # xtrain = standardization(xtrain)
    W1 = net[0]
    b1 = net[1]
    W2 = net[2]
    b2 = net[3]
    a1 = W1.dot(xtrain) + b1
    y1 = relu(a1)
    y2 = W2.dot(y1) + b2
    y = softmax(y2)
    return y

def eval_loss(y, d):
    i = np.shape(d)[0]
    j = np.shape(d)[1]
    di_log_yi = -d * np.log(y)
    E_e = di_log_yi.sum(axis = 1) / j
    E = E_e.sum(axis = 0) / i
#     return -(d * np.log(y)).mean()
    return E

def update_shallow(x, d, net, gamma):
    W1 = net[0]
    b1 = net[1]
    W2 = net[2]
    b2 = net[3]
    gamma = gamma / x.shape[0] # normalized by the training dataset size
    x = standardization(x)
    a1 = W1.dot(x) + b1
    h1 = relu(a1)
    a2 = W2.dot(h1) + b2
    y = softmax(a2)
    # e = (y - d)
    # print(d)
    e = -d/y + (1-d)/(1-y)

    delta2 = softmax_backp(a2, e)
    delta1 = relu_backp(a1, W2.T.dot(delta2))
    W2 = W2 - gamma * delta2.dot(h1.T)
    W1 = W1 - gamma * delta1.dot(x.T)
    b2 = b2 - gamma * delta2.sum(axis=1, keepdims=True)
    b1 = b1 - gamma * delta1.sum(axis=1, keepdims=True)
    return W1, b1, W2, b2

def backprop_minibatch_shallow(x, d, net, T, B = 8, gamma=.0001):
    N = x.shape[1]
    NB =int((N+B-1)/B)
    lbl = onehot2label(d)
    for t in range(T):
        shuffled_indices = np.random.permutation(range(N))
        for l in range(NB):
            minibatch_indices = shuffled_indices[B*l:min(B*(l+1), N)]
            x_mini = x[:, minibatch_indices]
            d_mini = d[:, minibatch_indices]
            net = update_shallow(x_mini, d_mini, net, gamma)
            y = forwardprop_shallow(x, net)
        loss = eval_loss(y, d)
        precision, recall = eval_perfs(y, lbl)
        print ("epoch = %d; eval_loss: %0.18f; eval_precision: %0.16f; eval_recall: %0.16f" % (t+1,loss, precision, recall))
    return net


def label2onehot(lbl):
    lbl = lbl.astype(int)
    d = np.zeros((lbl.max() + 1, lbl.size))
    d[lbl, np.arange(lbl.size)] = 1
    return d

def onehot2label(d):
    lbl = d.argmax(axis=0)
    return lbl

def eval_perfs(y, lbl, Type):
    # Use the following code for RF_Threshold
    if Type == 'RF_Threshold':
        prid_y = y
    # Use the following code for RF and KNN
    if Type == 'RF_Original':
        prid_y = onehot2label(y.T)
    # Use the follwing code for ANN
    if Type == 'ANN':
        prid_y = onehot2label(y)
    print("\nPridiction Y value is {} using {} model:".format(prid_y, Type))
    print("Ground Truth Y value is:", lbl)
    true_positive = 0
    false_negative = 0
    false_postive = 0
    true_negative = 0
    total_count = 0
    precision = 0
    for i in range(np.shape(lbl)[0]):
        if lbl[i] == prid_y[i]:
            if lbl[i] != 0:
                true_positive += 1
                total_count += 1
            else:
                true_negative += 1
                total_count += 1
        else:
            if lbl[i] == 0:
                false_postive += 1
                total_count += 1
            elif lbl[i] == 1:
                false_negative += 1
                total_count += 1
            elif lbl[i] == 2:
                false_negative += 1
                total_count += 1
            else:
                false_negative += 1
                total_count += 1

    if (true_positive + false_postive) != 0:
        precision = true_positive / (true_positive + false_postive)
    recall = true_positive / (true_positive + false_negative)
    # print(true_positive)
    return precision, recall


y_train = label2onehot(np.array(y_train))
lbl_train = onehot2label(y_train)
print('lable train nonzero numbers', np.count_nonzero(lbl_train))
y_test = label2onehot(np.array(y_test))

Ni = x_train.T.shape[0]  #number of features
Nh = 12  # number of hidden layer
No = y_train.shape[0]  #number of labels

# ANN_Model_Path = 'ANN_Model.npy'
# if not os.path.exists(ANN_Model_Path):
#     Net_Init = init_shallow(Ni, Nh, No)
#     netminibatch = backprop_minibatch_shallow(x_train.T, y_train, Net_Init, 10)
#     np.save(ANN_Model_Path, netminibatch)
#     print("ANN Model File is Created")
# else:
#     ANN_Net_Model = np.load(ANN_Model_Path)
#
# # yinit = forwardprop_shallow(x_train.T, Net_Init)
# netminibatch = backprop_minibatch_shallow(x_train.T, y_train, ANN_Net_Model, 10)
# np.save(ANN_Model_Path, netminibatch)
#
# predict_y_test = forwardprop_shallow(x_test.T, netminibatch)
# lbl_test = onehot2label(y_test)
# print("Ground Truth lbl nonzero numbers", np.count_nonzero(lbl_test))
#
# precision, recall = eval_perfs(predict_y_test, lbl_test)
# print("eval_precision: %0.16f; eval_recall: %0.16f" % (precision, recall))


# # RF Model Detail Showing as Following
clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                             random_state=0)

lbl_train = onehot2label(y_train)
lbl_test = onehot2label(y_test)

print("Training Dataset Input shape is ", x_train.shape)
print("Training Dataset Output shape is ", y_train.shape)
print("Testing Dataset Input shape is ", x_test.shape)

random_forest_model = clf.fit(x_train, y_train.T)

prediction_output = random_forest_model.predict(x_test)

precision, recall = eval_perfs(prediction_output, lbl_test, Type='RF_Original')

print("Test Data Precision using RF_Original is:", precision)
print("Test Data Recall using RF_Original is:", recall)

RF_Model_Path = 'RF_Model.npy'
filename = 'RF_model.sav'
if not os.path.exists(RF_Model_Path):
    # save rf model with pickle
    pickle.dump(random_forest_model, open(filename, 'wb'))
    np.save(RF_Model_Path, random_forest_model)
    print("RF Model File is Created")

test_rf_model = pickle.load(open(filename, 'rb'))
print("RF Model File is Loaded")

output = random_forest_model.predict(x1_test)

label = onehot2label(output.T)

prob_output = random_forest_model.predict_proba(x1_test)

prob_output = np.array(prob_output)

# probability matric: row shows the n-th class (total 4 classes in this project) and column shows the N-th number of sample testing data
# Probability output 0 index is the n-th number of class
# Probability output 1 index is the n-th number of sample
# Probability output 2 index is the 0-1 probability for yes or no

Class_Dict = {0: 'Normal Swimming', 1: 'Distress', 2: 'Risk', 3: 'Drowning'}
Threshold = 0.85
prediction_output_prob =  random_forest_model.predict_proba(x_test)#Used for Test Probability
prob_output = prediction_output_prob
prob_output = np.array(prob_output)

New_Y_Pre_Threshold = []

for s in range(prob_output.shape[1]):
    class_prob_list = []
    for c in range(prob_output.shape[0]):
        # print('Sample test {} has the probability of class {} is {}'.format(s, Class_Dict[c], prob_output[c, s, 1]))
        class_prob_list.append(prob_output[c, s, 1])
    #Can Set Up Threshold Right Here
    if class_prob_list[0] < Threshold:
        class_prob_list[0] = 0
    # if Class_Dict[class_prob_list.index(max(class_prob_list))] != Class_Dict[lbl_test[s]]:
    #     print(class_prob_list[0])
    print("According to the RF prediction, number {} sample dataset should be class {}".format(s, Class_Dict[class_prob_list.index(max(class_prob_list))]))
    print("The Ground Truth value is {}\n".format(Class_Dict[lbl_test[s]]))
    New_Y_Pre_Threshold.append(class_prob_list.index(max(class_prob_list)))


precision, recall = eval_perfs(prediction_output, lbl_test, Type='RF_Original')

print("\nTest Data Precision using RF_Original is:", precision)
print("Test Data Recall using RF_Original is:", recall)

new_precision, new_recall = eval_perfs(New_Y_Pre_Threshold, lbl_test, Type='RF_Threshold')

print("\nTest Data Precision using RF with Threshold is:", new_precision)
print("Test Data Recall using RF with Threshould is:", new_recall)

# KNN Model Detail Showing as Following
# from sklearn.neighbors import KNeighborsClassifier
#
# knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
#
# lbl_train = onehot2label(y_train)
# lbl_test = onehot2label(y_test)
#
# print("Training Dataset Input shape is ", x_train.shape)
# print("Training Dataset Output shape is ", y_train.shape)
# print("Testing Dataset Input shape is ", x_test.shape)
#
# knn_model = knn_clf.fit(x_train, y_train.T)
#
# prediction_output = knn_model.predict(x_test)
#
# precision, recall = eval_perfs(prediction_output, lbl_test, Type='RF_Original')
#
# print("Test Data Precision using KNN is:", precision)
# print("Test Data Recall using KNN is:", recall)

