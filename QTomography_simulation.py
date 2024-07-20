import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from time import sleep

def randListGen(leng):
    randlist = []
    for i in range(leng):
        randlist.append(random.randint(0, 1))
        sleep(0.01)
    nprandlist = np.array(randlist)
    return nprandlist


def translate_H(arr):
    arrtransH = np.zeros(arr.shape)
    for i in range(0, len(arr)):
        if not arr[i]:
            arrtransH[i] = 1
    return arrtransH

def translate_V(arr):
    arrtransV = np.zeros(arr.shape)
    for i in range(0, len(arr)):
        if arr[i]:
            arrtransV[i] = 1
    return arrtransV


# Creating HH-VV base simulated results
randomBits_list1 = randListGen(1000)
randomBits_list1 = np.array(randomBits_list1)
HH_VV_base = np.zeros((1000, 5))
HH_VV_base[:, 0] = randomBits_list1
HH_VV_base[:, 1] = translate_H(randomBits_list1)
HH_VV_base[:, 2] = translate_V(randomBits_list1)
HH_VV_base[:, 3] = translate_H(randomBits_list1)
HH_VV_base[:, 4] = translate_V(randomBits_list1)
print(HH_VV_base)
dfHH_VV = pd.DataFrame(HH_VV_base)
columns = ['bit', 'Alice-H', 'Alice-V', 'Bob-H', 'Bob-V']
dfHH_VV.set_axis(columns, axis=1, inplace=True)
print(dfHH_VV)
dfHH_VV.to_csv(r'C:\Users\yuval\Documents\Physics lab C2\HH-VV simulation.csv', index=False)

# Creating HV-VH base simulated results
randomBits_list2 = randListGen(1000)
randomBits_list2 = np.array(randomBits_list2)
HV_VH_base = np.zeros((1000, 5))
HV_VH_base[:, 0] = randomBits_list2
HV_VH_base[:, 1] = translate_H(randomBits_list2)
HV_VH_base[:, 2] = translate_V(randomBits_list2)
nots = np.logical_not(randomBits_list2)
HV_VH_base[:, 3] = translate_H(nots)
HV_VH_base[:, 4] = translate_V(nots)
print(HV_VH_base)
dfHV_VH = pd.DataFrame(HV_VH_base)
columns = ['bit', 'Alice-H', 'Alice-V', 'Bob-H', 'Bob-V']
dfHV_VH.set_axis(columns, axis=1, inplace=True)
print(dfHV_VH)
dfHV_VH.to_csv(r'C:\Users\yuval\Documents\Physics lab C2\HV-VH simulation.csv', index=False)

