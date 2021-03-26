import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import random

logs = open(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), 'BadActorLogs.txt'), 'r', encoding="utf8") #Opens logs.txt (has to be in same directory as python script)

#training labels for combined logs
#trainingLabels = ["safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","unsafe","safe","safe","unsafe","unsafe","unsafe","safe","safe","safe","unsafe","unsafe","unsafe","safe","unsafe","unsafe","unsafe"]

#training labels for bad actor logs
trainingLabels = ["Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","failed login","failed login","failed login","unauthorized login","unauthorized login","unauthorized login"]

# Vectorizes log file into numerical array
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(logs)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
for i in range(len(denselist)):
        denselist[i].insert(0,trainingLabels[i])

random.shuffle(denselist)

df = pd.DataFrame(denselist)

#import data
balance_data = df


#NOTHING WORKING YET, JUST THROWING CODE AT A WALL FOR SPAGHETTI CODE (NOT AL DENTE!)
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import seaborn as sns

def Driver():
   logRegrModel = LogisticRegression()
   logRegrModel.fit(x_train, y_train)
   rfe = RFE(logreg, 20)
   rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
   print(rfe.support_)
   print(rfe.ranking_)

if __name__=='__main__':
    Driver()
