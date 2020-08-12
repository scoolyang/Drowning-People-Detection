import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

class Li_Yang_Model(object):
    def __init__(self, input_data, saved_model_filename = 'RF_model.sav', threshold = 0.85, model_type = 'rf_model'):
        self._model_type = model_type
        self._input_data = input_data
        self._filename = saved_model_filename
        self.threshold = threshold

    def load_model(self):
        if self._model_type == 'rf_model':
            filename = self._filename
            loaded_model = pickle.load(open(filename, 'rb'))

        return loaded_model

    def prediction(self):
        Class_Dict = {0: 'Normal Swimming', 1: 'Distress', 2: 'Risk', 3: 'Drowning'}
        cur_model = self.load_model()
        std_input_data = self.standardization()
        prediction_output_prob = cur_model.predict_proba(std_input_data)  # Used for Test Probability
        prob_output = np.array(prediction_output_prob)
        New_Y_Pre_Threshold = []
        for s in range(prob_output.shape[1]):
            class_prob_list = []
            for c in range(prob_output.shape[0]):
                class_prob_list.append(prob_output[c, s, 1])
            if class_prob_list[0] < self.threshold:
                class_prob_list[0] = 0
            New_Y_Pre_Threshold.append(class_prob_list.index(max(class_prob_list)))

        return New_Y_Pre_Threshold

    def standardization(self):
        train_mean = np.load('Train_Mean.npy')
        train_std = np.load('Train_Std.npy')
        test_std_data = (self._input_data - train_mean) / train_std

        return test_std_data



# if __name__ == '__main__':
    # x1 = np.array([[1.7223130e-01, 6.1931826e+01, 7.4554722e-02,
    #                6.7288089e-01, 6.8041544e-02, 8.6643994e-01],
    #               [3.8890055e-02, 2.1595719e+01, 7.7176080e-03,
    #                7.5837305e-02, 1.0487424e-01, 4.2562320e-02],
    #               [2.2666759e-01, 7.0483095e+01, 7.1463925e-02,
    #                6.3008836e-02, 5.5187415e-02, 3.9298553e-02],
    #               [1.0625489e-01, 5.5562917e-01, 9.3402372e-02,
    #                6.3901003e-03, 1.0420951e-01, 2.1379018e-02],
    #               [1.3226214e-01, 1.8500301e+01, 7.1754457e-02,
    #                6.6085574e-02, 7.9599375e-02, 1.6758212e-02]]).reshape(5, 6)
    # model = Li_Yang_Model(x1)
    # result = model.prediction()

    # print(result)