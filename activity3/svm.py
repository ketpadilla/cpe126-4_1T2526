'''
  #Full Name: KRISTIAN EMMANUEL T. PADILLA

  #Course-Section: CPE126-4-E02

  #Credits to: https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg ; "Tech With Tim
'''

'''
  Main Code
'''

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# load dataset
cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)

# set features and target 
X = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
# print(x_train, y_train)

# create model
classes = ['malignant', 'benign']
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

# test model
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)