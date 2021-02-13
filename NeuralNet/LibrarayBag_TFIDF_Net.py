import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
logs = open('S:/Logs/allLogs.txt', 'r', encoding="utf8")
#import re

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(logs)
logSet = np.array(vectors.toarray())
#labels for our set of logs
logLabels = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,0,1,1,1])

""" labels for school secure logs
logLabels = np.zeros(len(logSet), dtype=int)
badActors = [7,8,9,73,74,75,76,77,77,78,78,2345,2346,2347,2348,2348,2349,2349,2601,2602,2603,2604,2605,2606,2607,3061,3062,3063,3064,3065,3066,3067,3068,3069,3070,3071,3072,3073,4625,4626,4627,4628,4629,4630,4631,4632,4633,4634,4635,4636,4637,5355,5356,5357,5358,5359,5360,5361,5362,5363,5364,5365,5366,5367,5368,9294,9295,
             9296,9297,9298,9299,9300,9301,9302,9303,9304,9305,9306,9307,9308,9309,9310,9311,12442,12443,12443,1344,13450,13451,13452,13453,13454,13455,13456,13457,13458,13459,13460,13461,15109,15110,15111,15112,15113,15114,15115,15116,15117,15118,15119,15120,16946,16947,16948,16949,16950,16950,16951,16951,16952,16953,16954,16955,16956,
             16957,17288,19007,19008,19009,19010,19011,19012,19013,19014,19015,19016,19017,19018,19019,20302,21089,21090,21091,21092,21093,21094,21095,21096,21097,21098,21099,23125,23126,23127,23128,23129,23130,23131,23132,23133,23134,23135,23136,24128,24998,24999,25000,25001,25002,25003,25004,25005,25005,25006,25006,
             25669,26702,26703,26704,26705,26706,26707,26708,26709,26710,26711,28758,28759,28760,28761,28762,28763,28764,28765,28766,29571,30917,31057,31058,31059,31060,31061,31062,31063,31064,31065,31066,31067,31634,32514,32515,32516,32517,32518,32519,32520,32521,33057,35345,35346,35347,35348,35349,35350,
             35351,35352,35353,35354,35355,36423,36447,36838,36839,36840,36841,36842,36843,36844,36845,37185,37437,37487,37488,37489,37490,37491,37492,37493,37494,37495,37496,37497,37498,37499,37500,37501,37502,37942,37943,37944,37945,37946,37947,37948,37949,37950,37951,37952,38139,38148,38330,38331,38332,38333,38334,38335,38336,38337,38338,38339,38340,4030,40302,
             40303,40304,40305,40306,40307,40308,40309,40310,40311,40312,40313,41107,41116,41806,41955,42421,42422,42423,42424,42425,42426,42427,42428,42429,43157,43605,43920,43973,43974,43975,43976,43977,43978,43979,43980,43981,44065,44066,44067,44067,44068,44068,44069,44070,44071,44072,44169,45581,45582,45583,45828,46021,46022,46023,46024,46025,46026,46027,46028]
for i in badActors:
    logLabels[i-1] = 1
"""

#Shuffles logs and labels in the same way
randomize = np.arange(len(logSet))
np.random.shuffle(randomize)
logSet = logSet[randomize]
logLabels = logLabels[randomize]

#Splits into training and testing data
trainingLogs = np.array(logSet[0:math.floor(len(logSet)/2)])
testLogs = np.array(logSet[math.floor(len(logSet)/2):len(logSet)])
trainingLabels = np.array(logLabels[0:math.floor(len(logLabels)/2)])
testLabels = np.array(logLabels[math.floor(len(logLabels)/2):len(logLabels)])




model = keras.Sequential([
                          keras.layers.Flatten(input_shape = (len(logSet[1]),)), #Number of features input, equal to length of dictionary
                          keras.layers.Dense(64, activation = "relu"), #I believe this is the number of nodes in the middle layer, don't quote me on that
                          keras.layers.Dense(2,activation="softmax") # number of possible labels
                          ])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(trainingLogs, trainingLabels, epochs=10) #epochs is number of times it will run through the data, 
                                                 
testLoss, testAcc = model.evaluate(testLogs, testLabels)
prediction = model.predict(testLogs)

for i in range (len(testLabels)):
   # if testLabels[i] != np.argmax(prediction[i]):
        print('Actual: ' , testLabels[i] , '  Prediction: ' , np.argmax(prediction[i]))



