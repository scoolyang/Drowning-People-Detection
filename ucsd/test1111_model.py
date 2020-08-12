import xiaozhuzhu
import numpy as np

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
model = xiaozhuzhu.Li_Yang_Model(x1)
result = model.prediction()

print(result)