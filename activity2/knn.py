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
print(car_evaluation.metadata) 
  
# variable information 
print(car_evaluation.variables) 
