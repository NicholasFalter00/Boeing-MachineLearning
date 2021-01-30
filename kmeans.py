import re
#import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

logs = open('School_Logs/ANON-secure-20210117', 'r') #TODO: INSERT LOCATION OF LOG FILE HERE
dictionary = {}
baseLog = [] #Will be added to as dictionary is populated. Has a 0 for each dictionary member.
logSet = []
baggedLogSet = []
tf = {}
idf = {}
#def inDictionary(feature):
#    for member in dictionary:
#        if member == feature:
#            return 1
#    return 0

"""
#this method counts the number of occurences of features in the entire file (because baggedLog and baseLog refer to the same thing. This is poor code)
#TODO: Remove or improve this function.
def bag(log): #implement bagofwords
    baggedLog = baseLog #start with a vector of 0's (n zeros, n dictionary members)
    for feature in log: #look at each feature in the log
        count = 0 #keep track of which member of the dictionary is being compared
        inDictionary = 0 #represents if the feature is found in the dictionary
        for member in dictionary: #search through dictionary
            if member == feature: #if feature is in dictionary
                baggedLog[count] += 1 #add 1 to the frequency of the feature
                inDictionary = 1
                break #exit search for feature in dictionary
            count += 1
        if not inDictionary: #if feature isn't in dictionary
            dictionary.append(feature) #add it to the dictionary
            baseLog.append(1) #add a zero for the new feature
    return baggedLog
"""

def bag(log): #implement bagofwords
    baggedLog = baseLog.copy() #start with a vector of 0's (n zeros, n dictionary members)
    for feature in log: #look at each feature in the log
        count = 0 #keep track of which member of the dictionary is being compared
        inDictionary = 0 #represents if the feature is found in the dictionary
        for member in dictionary: #search through dictionary
            if member == feature: #if feature is in dictionary
                baggedLog[count] += (1/len(log)) #add 1 to the frequency of the feature
                inDictionary = 1
                dictionary.update({member: dictionary.get(member) + 1})
                break #exit search for feature in dictionary
            count += 1
        if not inDictionary: #if feature isn't in dictionary
            dictionary.update({feature: 1}) #add it to the dictionary
            baseLog.append(0) #add a zero for the new feature
            baggedLog.append((1/len(log))) #Count the new feature
    return baggedLog

for line in logs:
    #logSet.append(line)
    if re.search("^\[",line): #log starts with a square bracket, indicating this is an error.log log
        #print('************Error LOG*************')
        line = re.split("\[", line, 1)[1]
        date = re.split("(?<=.{10})\s", line, 1)[0] #gets first part of date. Will be concatenated with year later
        line = re.split("(?<=.{10})\s", line, 1)[1]
        time = re.split("\s", line, 1)[0]
        #print("time = " + time)
        line = re.split("\s", line, 1)[1]
        date = date + " " + re.split("]", line, 1)[0] #concatenates year with the rest of the date
        #print("date = " + date)
        line = re.split("]", line, 1)[1]
        line = re.split("\[", line, 1) [1]
        logType = re.split("\]", line, 1) [0]
        #print("log type = " + logType)
        line = re.split("\[", line, 1) [1]
        IDs = re.split("\]", line, 1) [0]
        PID = re.split("\s", IDs, 1)[1]
        PID = re.split("\:", PID, 1)[0]
        #print("PID = " + PID)
        TID = re.split("\s", IDs, 2)[2]
        #print("TID = " + TID)
        if re.search("\[(?=client)", line): #ignore if brackets arent for client
            line = re.split("\[", line, 1)[1]
            line = re.split("\s", line, 1)[1]
            client = re.split("\]", line, 1)[0]
            #print("client = " + client)
        line = re.split("\s(?=A)", line, 1)[1] #find space followed by "A" 
        errorCode = re.split("\:", line, 1)[0]
        #print("error code = " + errorCode)
        msg = re.split("\:", line, 1)[1]
        msg = re.split("\t", msg, 1)[0]
        #print("message =" + msg, end = "" ) #TODO: remove \n from the end of msg
        log = [time, date, logType, errorCode, msg]
        baggedLog = bag(log)
        del log
        #print("bagged log: ", end = "")
        #print(baggedLog)
        baggedLogSet.append(baggedLog)
        #print("")
        
    elif re.search("^\d",line): #log starts with a digit, indicating this is an access.log log
        #used for malicious web server access bad actor
        #print('************ACCESS LOG*************')
        ip = re.split("\s", line, 1)[0]
        #print("ip = " + ip)
        line = re.split("\s", line, 1)[1]
        line = re.split("(?<=-)\s", line, 1)[1]
        user = re.split("\s", line, 1)[0]
        #print("user = " + user)
        line = re.split("\s\[", line, 1)[1]
        date = re.split(":", line, 1)[0]
        #print("date = " + date)
        line = re.split(":", line, 1)[1]
        time = re.split("]", line, 1)[0]
        #print("time = " + time)
        msg = re.split("] ", line, 1)[1]
        msg = re.split("\t", msg, 1)[0]
        #print("message =" + msg, end = "" ) #TODO: remove \n from the end of msg
        log = [ip, user, date, time, msg]
        baggedLog = bag(log)
        del log
        #print("bagged log: ", end = "")
        #print(baggedLog)
        baggedLogSet.append(baggedLog)
        #print("")

    elif re.search("^[A-Za-z]",line): #log starts with a alphabetical character, indicating this is an auth.log log
        #print('************AUTH LOG*************')
        date = re.split("(?<=.{6})\s", line, 1)[0]
        #print("date = " + date)
        line = re.split("(?<=.{6})\s", line, 1)[1]
        time = re.split("(?<=.{8})\s", line, 1)[0]
        #print("time = " + time)
        line = re.split("(?<=.{8})\s", line, 1)[1]
        user = re.split("\s", line)[0]
        #print("user = " + user)
        msg = re.split("\s|\t", line, 1)[1]
        msg = re.split("\t", msg, 1)[0]
        #print("message =" + msg, end = "" ) #TODO: remove \n from the end of msg
        log = [date, time, user, msg]
        baggedLog = bag(log)
        del log
        #print("bagged log: ", end = "")
        #print(baggedLog)
        baggedLogSet.append(baggedLog)
        #print("")

logs.close()

#print("Dictionary: ")
#print(dictionary)

maxLen = len(baggedLogSet[-1])
for i in range (len(baggedLogSet)): # make all subarrays same length
    baggedLogSet[i].extend([0]*(maxLen - len(baggedLogSet[i])))

#print(len(baggedLogSet))

val = list(dictionary.values())

for i in range(len(baggedLogSet)):
    for j in range(len(baggedLogSet[i])):
        baggedLogSet[i][j] = baggedLogSet[i][j] * np.log(len(baggedLogSet)/val[j])

#print(baggedLogSet)

Kmean = KMeans(n_clusters=8)
Kmean.fit(np.array(baggedLogSet))

#print(Kmean.cluster_centers_)
print(Kmean.labels_)

#print(len(Kmean.labels_))

clusterLabels = list(Kmean.labels_)

#for i in range(8):
    #print("\n" + str(i) + "\n")
    #for j in range(len(clusterLabels)):
        #if(clusterLabels[j] == i):
            #print(logSet[j])

print("done")