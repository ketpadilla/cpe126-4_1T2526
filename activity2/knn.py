'''
  #Full Name: KRISTIAN EMMANUEL T. PADILLA

  #Course-Section: CPE126-4-E02

  #Credits to: https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg ; "Tech With Tim
'''

'''
    UCI Import Code

    Bohanec, M. (1988). Car Evaluation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48.
'''

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
# metadata 
# print(car_evaluation.metadata) 
  
# # variable information 
# print(car_evaluation.variables) 

'''
  Main Code
'''

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier  
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# data
data = pd.concat([X, y], axis=1)
print(data.head())

# preprocessing
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# separate features and target
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
  print("Predicted: ", names[predicted[x]], "Data: ", [int(v) for v in x_test[x]], "Actual: ", names[y_test[x]])

  n = model.kneighbors([x_test[x]], 9, True)
  print("N: ", n)