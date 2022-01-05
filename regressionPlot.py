# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:21:53 2021

@author: robyn
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fileName = 'C:\\Users\\robyn\\Documents\\Grad_School\\Research\\swim-python\\Inputs\\GrandEvapRegression.xlsm'

Sheet = pd.read_excel(fileName, sheet_name='2021-2070', index_col=0)

# col 5 = 'ET-ETavg', col 6 = 'T-Tavg'
regrET = Sheet['modeled_ET']
crleET = Sheet['LakeET_mm']
sns.scatterplot(x=crleET, y=regrET, alpha=.4)
plt.ylabel('Regression Model Annual Evaporation (mm/year)')
plt.xlabel('CRLE Model Annual Evaporation (mm/year)')

#add line representing linear releationship
a = [1250,1600]
b = [1250,1600]
plt.plot(a,b)

arrayCRLE = []
arrayRegr = []

for i in range(49,5599,50):
    arrayCRLE.append(regrET.iloc[i-50:i].mean())
    arrayRegr.append(crleET.iloc[i-50:i].mean())

plt.figure()
sns.scatterplot(x=arrayCRLE, y=arrayRegr, data=Sheet,alpha=.4)
plt.ylabel('Regression Model Avg-Annual Evaporation (mm/year)')
plt.xlabel('CRLE Model Avg-Annual Evaporation (mm/year)')
plt.plot(a,b)  
    

    


