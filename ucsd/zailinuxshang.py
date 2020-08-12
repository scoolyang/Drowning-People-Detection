import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
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


#Feature preprocessing if needed

drop_column = ['Index', 'Output']
X_feature_space = data_frame.drop(drop_column, axis=1)


#Get ground_truth and apply one-hot coding
y_ground_truth = data_frame["Output"]

def standardization_test(train_matrix, test_matrix):
    x_train_matrix = np.array(train_matrix)
    x_test_matrix = np.array(test_matrix)
    x_mean = x_train_matrix.sum(axis = 0) / x_train_matrix.shape[0]
    x_std = np.sqrt(((x_train_matrix-x_mean)**2).sum(axis=0) / x_train_matrix.shape[0])
    if not os.path.exists(Train_Mean_Path) or not os.path.exists(Train_Std_Path):
        np.save(Train_Mean_Path, x_mean)
        np.save(Train_Std_Path, x_std)
    print('train data mean value is,\n', x_mean)
    print('train data std value is, \n', x_std)
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

output = test_rf_model.predict(x1_test)

label = onehot2label(output.T)

prob_output = test_rf_model.predict_proba(x1_test)

prob_output = np.array(prob_output)

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
    # print("According to the RF prediction, number {} sample dataset should be class {}".format(s, Class_Dict[class_prob_list.index(max(class_prob_list))]))
    # print("The Ground Truth value is {}\n".format(Class_Dict[lbl_test[s]]))
    New_Y_Pre_Threshold.append(class_prob_list.index(max(class_prob_list)))


precision, recall = eval_perfs(prediction_output, lbl_test, Type='RF_Original')

print("\nTest Data Precision using RF_Original is:", precision)
print("Test Data Recall using RF_Original is:", recall)

new_precision, new_recall = eval_perfs(New_Y_Pre_Threshold, lbl_test, Type='RF_Threshold')

print("\nTest Data Precision using RF with Threshold is:", new_precision)
print("Test Data Recall using RF with Threshould is:", new_recall)

