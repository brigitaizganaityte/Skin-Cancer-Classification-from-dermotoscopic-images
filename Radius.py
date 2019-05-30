import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import math
from math import sqrt
from sklearn.model_selection import train_test_split

#reading csv file of ImageJ histogram results
res = []
with open("histogramu_rezultatai_2.csv", "r") as inputFile:
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
data_df.columns = ['labels', 'component1', 'component2',
                   'component3']
x1 = (np.mean(data_df['component1']))
y1 = (np.mean(data_df['component2']))
z1 = (np.mean(data_df['component3']))
dist = []

#distances of every point to the center point
for i in range(0, len(data_df)):
	dist.append(sqrt((data_df.iloc[i]['component1'] - x1)**2+ 
	           (data_df.iloc[i]['component2']-y1)**2+ 
			   (data_df.iloc[i]['component3']-z1)**2))

#dataframe with distances and labels
distance = np.array(dist)
df_dist = pd.DataFrame(distance)
df_dist["labels"]=label
df_dist.columns = ['distances', 'labels']


#new dataframe for melonoma 
for j in range(0, len(df_dist)):
	mel_df = df_dist[df_dist['labels'] == '4'] 
#new dataframe for other labels
for j in range(0, len(df_dist)):
	other_df = df_dist[df_dist['labels'] != '4']
length = len(mel_df)
other_df = other_df[:length]
#dataframe that contains the same number of records 
#with other labels as melonoma
result_df = mel_df.merge(other_df, how='outer')

#histogram of benign and malignant cancer data 
mel_df.hist(grid=False, color = 'red')
plt.title('Duomenų išsidėstymas pagal spindulį')
other_df.hist(grid=False, color = 'green')
plt.title('Duomenų išsidėstymas pagal spindulį')
plt.show()

#spliting data for training and testing
train_df, test_df = train_test_split(result_df, test_size=0.2)

#Training
min_dist_tr = train_df['distances'].min()
max_dist_tr = train_df['distances'].max()
R_15_tr = np.arange(min_dist_tr,max_dist_tr, 1.50)
R_11_tr = np.arange(min_dist_tr,max_dist_tr, 1.10)
R_7_tr = np.arange(min_dist_tr,max_dist_tr, 0.7)
R_3_tr = np.arange(min_dist_tr,max_dist_tr, 0.3)

#new dataframe for selected labels
def new_dataframe(data_df, melonoma_df, lb):
	for j in range(0, len(data_df)):
		selected_df = data_df[data_df['labels'] == lb]
	length = len(selected_df)
	melonoma_df = melonoma_df[:length]
	res_select_df = selected_df.merge(melonoma_df, how='outer')
	return res_select_df
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
			if ((float(df_dist.iloc[j]['distances']) <= R[i]) & (
			    df_dist.iloc[j]['labels']=='4')):
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
	print ('Accurancy of the model when saying that melonoma data is inside R', 
	       (sum_inside_R_4[max_index]/(sum_inside_R_4[max_index]+
		    sum_inside_R_oth[max_index])))
	print ('Biggest radius that you can find for data points: ',
	       R[difference_for_min])

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
		if (old_labels[i] == '4'):
			match.append('True')
	print ('Matched: ',len(match))
	#sum of data that is under that radius
	print ('Out of: ',len(old_labels))
	print ('Model accurancy : ',(len(match)/(len(old_labels))))
	
#additional parameter lab-number of label number that best radius
#would be searching
def find_best_R_selected_label(R, df_dist, test_df,lab):
	inside_R_4 = []
	inside_R_oth = []
	
	sum_inside_R_4 = []
	sum_inside_R_oth = []
	
	fraction = []
	for i in range(0, len(R)): 
		for j in range(0, len(df_dist)):
			if ((float(df_dist.iloc[j]['distances']) <= R[i]) & 
			   (df_dist.iloc[j]['labels']==lab)):
				inside_R_oth.append(float(df_dist.iloc[j]['distances']))			
			elif ((float(df_dist.iloc[j]['distances']) <= R[i]) & 
			     (df_dist.iloc[j]['labels']=='4')):
				inside_R_4.append(float(df_dist.iloc[j]['distances']))		
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
	print ('Sum of selected label data inside R:', sum_inside_R_oth[max_index])
	print ('Accurancy of the model when saying that melonoma data is inside R', 
	       (sum_inside_R_4[max_index]/(sum_inside_R_4[max_index]+
		    sum_inside_R_oth[max_index])))
	print ('Biggest radius that you can find data points: ',
           R[difference_for_min])
	
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
		if (old_labels[i] == '4'):
			match.append('True')
		else:
			match.append('False')
	print ('Matched: ',len(match))
	#sum of data that is under that radius
	print ('Out of: ',len(old_labels))
	print ('Model accurancy : ',(len(match)/(len(old_labels))))
	

find_best_R(R_15_tr, train_df, test_df)

