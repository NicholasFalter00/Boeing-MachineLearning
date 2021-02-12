import re
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import math
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



logs = open('logs2.txt', 'r') #Opens logs.txt (has to be in same directory as python script)

# Vectorizes log file into numerical array
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(logs)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
#print(df)



#Error.log logs (for unauthorized web server logins bad actor and malicious webserver access)
trainingLabelsError = ["safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access"]
#Access.log logs (for DDOS)
trainingLabelsAccess = ["safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos"]
#Auth.log logs (for everything else)
trainingLabelsAuth = [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1]

#trainingLabelsError.extend(trainingLabelsAccess)

trainLabels = []

for label in trainingLabelsAccess:
    if label == 'safe':
        trainLabels.append(0)
    else:
        trainLabels.append(1)

trainLabels.extend(trainingLabelsAuth)

trainLabels.extend([0]*(9634 - len(trainLabels)))

#print(trainLabels)
#print(len(trainLabels))

model = GaussianNB()

model.fit(df.to_numpy(), trainLabels)

logs2 = open('logs.txt','r')

vectors2 = vectorizer.transform(logs2)
feature_names2 = vectorizer.get_feature_names()
dense2 = vectors2.todense()
denselist2 = dense2.tolist()
df2 = pd.DataFrame(denselist2, columns=feature_names2)

#print(df2)

predicted = model.predict(df2.to_numpy())
#print(predicted)
predictedProb = model.predict_proba(df.to_numpy())
print(predictedProb)

accuracy_score()

