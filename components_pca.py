import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = []
#for splitting given data
def chunks(st, numb):
    for start in range(0, len(st), numb):
        yield st[start:start+numb] 
#reading csv file that contains x,y values
#of objects edges and appending data to data array
with open('C:/Users/Brigita/Desktop/Bakalaurinis/res_pca.csv')as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        data.append(row[0])
#reading csv file that contains given data labels
#and appending it to labels array
labels = []
with open('C:/Users/Brigita/Desktop/Bakalaurinis/labels.csv') as file:
    csvReader2 = csv.reader(file)
    for row in csvReader2:
        labels.append(row[0])
#appending data of each image to a different array. 
#4200 - pixels of each image projections 1200 (first and third) 
#and + 900 (second and fourth)		
image_data = []
for i in chunks(data,4200):
    image_data.append(i)
#creating dataframe : each row for each picture and columns for pixels values:
#600 of X values of first projection, 600 of Y values of first projection
#450 of X values of second projection, 450 of Y values of second projection
#600 of X values of third projection, 600 of Y values of third projection
#450 of X values of fourth projection, 450 of Y values of fourth projection
all_dataframe=pd.DataFrame(image_data)
x = StandardScaler().fit_transform(all_dataframe)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
data_df = pd.DataFrame(principalComponents)
#appending fourth column for images labels
data_df[3] = labels
#changing dataframe structure labels, 
#principal component 1, principal component 2, principal component 3
data_df = pd.DataFrame(data_df.values[:,[3,0,1,2]]) 
#creating csv file from result dataframe
with open('C:/Users/Brigita/Desktop/Bakalaurinis/full_res_pca.csv', 'w') as f:
		data_df.to_csv(f, header=False, index=None)
