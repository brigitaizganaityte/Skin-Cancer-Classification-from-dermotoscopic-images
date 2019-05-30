import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#reading csv file
res = []
with open("histogramu_rezultatai.csv", "r") as inputFile:
	for line in inputFile:
		line = line.rstrip()
		res.append(line)
def chunks(st, numb):
    for start in range(0, len(st), numb):
        yield st[start+1:start+numb] 
def labels(st, numb):
    for start in range(0, len(st), numb):
        yield st[start:start+1] 

#results array
results = []
for i in chunks(res,257):
    results.append(i)
label = []
#labels of disease array
for i in labels(res,257):
    label.append(i)
label = np.array(label)
data = np.array(results)


data = StandardScaler().fit_transform(data) 
#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(data)
data_df = pd.DataFrame(principalComponents)
data_df["labels"]=label
#changing order of the dataframe
data_df = pd.DataFrame(data_df.values[:,[3,0,1,2]])
data_df.columns = ['labels', 'component1',
                   'component2', 'component3']
#number of rows for the xyz file
rows = data_df.labels.count() 

#writing dataframe to xyz file(appending)
file=open('C:/Users/Brigita/Desktop/Bakalaurinis/Hist_pca.xyz','a')
#first row of the file - number of records 
file.write(str(rows) + '\n')
#second row of the file - title
file.write("Histogram data" + '\n')
#writing results
np.savetxt(file, data_df.values, fmt='%s')
file.close()



