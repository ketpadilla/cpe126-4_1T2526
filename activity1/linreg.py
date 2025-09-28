'''
#Full Name: KRISTIAN EMMANUEL T. PADILLA

#Course-Section: CPE126-4-E02

#Credits to: https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg ; "Tech With Tim
'''

'''
  UCI Import Code

  Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
'''
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 

'''
  Main Code
'''
# imports
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# data
data = pd.concat([X, y], axis=1)

print(data.head())

# data selection
# e.g. 1: focus only on grades
# data = data[["G1", "G2", "G3"]]

# e.g. 2: include studytime and failures
# data = data[["studytime", "failures", "G3"]]

# e.g. 3: mix absences with grades
# data = data[["G1", "G2", "absences", "G3"]]

# e.g. 4: all numeric columns 
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age", "Medu", "Fedu"]]

predict = "G3"

# separate features and target
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# train and test
best = 0
for _ in range(30):

  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
  linear = linear_model.LinearRegression()

  linear.fit(x_train, y_train)
  acc = linear.score(x_test, y_test)
  print(acc)

  if acc > best:
    best = acc
    with open("./activity1/studentlinreg.pickle", "wb") as f:
      pickle.dump(linear, f)

pickle_in = open("./activity1/studentlinreg.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: \n",linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])

# plot
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()