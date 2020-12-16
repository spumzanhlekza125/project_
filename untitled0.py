# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:37:52 2020

@author: 216042447
"""
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from operator import itemgetter
from collections import Counter
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

def euclidean_distance(vector1, vector2):
	distance = 0.0
	for i in range(len(vector1)):
		distance += ((vector1[i] - vector2[i])**2)
	return math.sqrt(distance)

#calculating the shotest distance
def outlier(li,k):
    li.sort(key=lambda l:l[2])
    li1 = li[-k:]
    return li1

#get name of player and the didtance
def get_list(mean_list,list1,full_list):
    new_list = []
    for x in range(len(list1)):
        shot_list = []
        distance = euclidean_distance(mean_list, list1[x])
        shot_list.append(full_list[x][1])
        shot_list.append(full_list[x][2])
        shot_list.append(distance)
        new_list.append(shot_list)
    return new_list

sns.set(style="ticks")
#Import Data 
data = pd.read_csv(r'player_regular_season_career.csv',encoding = 'utf-8')
 
# Avoid the seaborn package Chinese display as a box question, change to English
print("name",data['lastname'])

data1 = np.array(data)
data_num = data.iloc[:, 4:-1]
data_num1 = np.array(data_num)
vec = data_num1[3435]
vec_mean = []
#print(euclidean_distance(vec, vec_mean))

for i in range(len(data_num1[1])):
    mean_vec = data_num.iloc[:,i:i+1]
    mean_vec = np.array(mean_vec)
    vec_mean.append(mean_vec.mean())

distance_list = get_list(vec_mean,data_num1,data1)
outli = outlier(distance_list,10)
for x in outli:
    print(x)
    
#print(get_list(vec_mean,data_num1,data1))
#print(len(data_num1))




















