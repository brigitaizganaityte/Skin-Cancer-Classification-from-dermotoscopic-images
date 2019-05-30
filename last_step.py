import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
from sklearn import metrics
from sklearn.model_selection import train_test_split
import math
from math import sqrt

#reading file of coordinates of detected objects edges
#header none - reads the first line of the file, too.
data_df = pd.read_csv('C:/Users/Brigita/Desktop/Bakalaurinis/full_res_pca.csv',
                      header = None)
#appending columns to dataframe
data_df.columns = ['labels','component1', 'component2', 'component3'] 

#new dataframe for melonoma 
for j in range(0, len(data_df)):
	mel_df = data_df[data_df['labels'] == 4] 
#new dataframe for other labels
for j in range(0, len(data_df)):
	other_df = data_df[data_df['labels'] != 4]
	
length = len(mel_df)
other_df = other_df[:length]
#dataframe that contains the same number of records 
#with other labels as melonoma
result_df = mel_df.merge(other_df, how='outer')
print (result_df)

data_without_labels = result_df[["component1", "component2", "component3"]]
data_without_labels = data_without_labels.as_matrix().astype("float32", 
                                                             copy = False)

data_without_labels = StandardScaler().fit_transform(data_without_labels)
#real diagnoses labels
labels_true = result_df['labels'].as_matrix().astype("int", copy = False)
#counting how much each label is in the dataframe that is given to DBSCAN
unique, sum = np.unique(labels_true, return_counts=True)

#DBSCAN
#eps - radius, min_samples - minimal number of record of cluster
db = cluster.DBSCAN(eps=0.5, min_samples=15).fit(data_without_labels)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# DBSCAN results
#-1 label for the outliers
clusters_numb = len(set(labels)) - (1 if -1 in labels else 0) 
outliers_numb = list(labels).count(-1) #-1 is the outlier for DBSCAN
#adding DBSCAN clusters labels to dataframe
result_df['cluster_label'] = labels

#finding data points number of DBSCAN created clusters
def inside_cluster(lab):
	real_label = []
	#given label of cluster and getting actual label of image
	for i in range(0, len(result_df)):
		if ((result_df.iloc[i]['cluster_label']==lab)):
			real_label.append((result_df.iloc[i]['labels']))
	real_label = np.array(real_label)	
	#number of unique elements in cluster
	unique_elem, sum_elem = np.unique(real_label, return_counts=True)
	print('Frequency of elements of', lab, 'cluster')
	print(np.asarray((unique_elem, sum_elem)))


#unique clusters labels
unique_labels = set(labels)
#getting information of each cluster
for i in unique_labels:
	inside_cluster(i)

print('DBSCAN results: number of clusters: %d' % clusters_numb)
print('DBSCAN results: number of outliers: %d' % outliers_numb)

#VISUALIZATION OF DBSCAN
#generating different colors for different diagnoses data points
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors): 
#-1 label is used for outliers and the dots are black
    if k == -1:
        col = [0, 0, 0, 1]	
    class_member_mask = (labels == k) 
   
    xy = data_without_labels[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = data_without_labels[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % clusters_numb)
plt.show()

#getting one cluster information
#-1 represents outliers and other numbers - different clusters
for i in range(0, len(result_df)):	
	klust_df = result_df[result_df['cluster_label'] == -1]
	
x1 = (np.mean(klust_df['component1']))
y1 = (np.mean(klust_df['component2']))
z1 = (np.mean(klust_df['component3']))

dist = []
#finding distances of data to center point
for i in range(0, len(klust_df)):
	dist.append(sqrt((klust_df.iloc[i]['component1'] - x1)**2 
	+ (klust_df.iloc[i]['component2']-y1)**2 
	+ (klust_df.iloc[i]['component3']-z1)**2))
klust_df['distances'] = dist

#spliting training and testing data
train_df, test_df = train_test_split(klust_df, test_size=0.2)

#Training
min_dist_tr = train_df['distances'].min()
max_dist_tr = train_df['distances'].max()
R_15_tr = np.arange(min_dist_tr,max_dist_tr, 1.50)
R_11_tr = np.arange(min_dist_tr,max_dist_tr, 1.10)
R_7_tr = np.arange(min_dist_tr,max_dist_tr, 0.7)
R_3_tr = np.arange(min_dist_tr,max_dist_tr, 0.3)

#searching for best sphere radius		
#parametres: R-steps of iteration, df_dist-data of distances and labels
#test_df - dataframe for testing (20%)
def find_best_R(R, df_dist, test_df):
	inside_R_4 = []
	inside_R_oth = []
	
	sum_inside_R_4 = []
	sum_inside_R_oth = []
	
	fraction = []
	for i in range(0, len(R)): 
		for j in range(0, len(df_dist)):
			if ((float(df_dist.iloc[j]['distances']) <= R[i]) & 
			   (df_dist.iloc[j]['labels']==4)):
				inside_R_4.append(float(df_dist.iloc[j]['distances']))		
			elif ((float(df_dist.iloc[j]['distances']) <= R[i])):			
				inside_R_oth.append(float(df_dist.iloc[j]['distances']))		
		sum_inside_R_4.append(len(inside_R_4))
		sum_inside_R_oth.append(len(inside_R_oth))		
		last_sum = int(sum_inside_R_4[-1])		
		inside_R_4 = []
		inside_R_oth = []		

	print ('Sum of melonoma labels data inside R: ', sum_inside_R_4)
	#for the first value subtraction, it should subtract 0
	for_diff_calc = np.insert(sum_inside_R_4, 0, 0) 
	#density of melonoma data inside R
	for i in range(0, len(sum_inside_R_4)):
		fraction.append(sum_inside_R_4[i]/sum_inside_R_4[-1])
	difference_for_min = np.argmax(fraction)
	difference = np.diff(for_diff_calc)	
	max_index = np.argmax(difference)	
	difference = difference[max_index:]
	min_index = np.argmin(difference) 

	print ('R for the biggest density of melonoma data', R[max_index]) 
	print ('Sum of melonoma data inside R:', sum_inside_R_4[max_index])
	print ('Sum of other label data inside R:', sum_inside_R_oth[max_index])
	print ('Accurancy of the model when saying melonoma data is inside R', 
	       (sum_inside_R_4[max_index]/(sum_inside_R_4[max_index]+
		    sum_inside_R_oth[max_index])))
	
	#Testing
	#biggest radius where you can find data
	RR = R[max_index]
	old_labels = []
	match = []
	print ('Testing')
	#unique values count in testing dataframe
	print ('Test dataframe:', test_df['labels'].value_counts())	
	for j in range(0, len(test_df)):
		if ((float(test_df.iloc[j]['distances']) <= float(RR))):
			old_labels.append((test_df.iloc[j]['labels']))
	for i in range(0, len(old_labels)):
		if (old_labels[i] == 4):
			match.append('True')
	print ('Matched: ',len(match))
	#sum of data that is under that radius
	print ('Out of: ',len(old_labels))
	print ('Model accurancy : ',(len(match)/(len(old_labels))))

find_best_R(R_15_tr, train_df, test_df)
