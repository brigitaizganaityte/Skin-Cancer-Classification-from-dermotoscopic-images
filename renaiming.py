import pandas as pd
import glob
import cv2 
import numpy as np

all_images=glob.glob('C:/Users/Brigita/Desktop/Bakalaurinis/HAM10000_images/*.jpg',
                       recursive=True)
data = []
#reading the data
for filename in all_images:
	im = cv2.imread(filename)
	im = cv2.resize(im, (50,50))
	im = np.array(im)
	im = np.concatenate(im) 
	im = np.concatenate(im)
	data.append(im)
	
#dictionary of diseases types
dictionary = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

df_metadata=pd.read_csv('C:/Users/Brigita/Desktop/Bakalaurinis/HAM10000_metadata.csv')
#new columns to metadata dataframe
df_metadata['cell_type'] = df_metadata['dx'].map(dictionary.get) 
df_metadata['cell_type_idx'] = pd.Categorical(df_metadata['cell_type']).codes
base_skin_dir = ('C:/Users/Brigita/Desktop/Bakalaurinis/HAM10000_images/')
#column with every full path to every image in the data
df_metadata['path'] = base_skin_dir+df_metadata["image_id"].astype(str)+".jpg"
#column how every image should be renaimed
df_metadata['rename']=(base_skin_dir+df_metadata["image_id"].astype(str)
                       +"_"+df_metadata['dx']+".jpg")
#list of actual names of the files in the directory
images_list = list(df_metadata['path'])
##list of new names of the files in the directory
images_new_list = list(df_metadata['rename'])

import os
for i in range(0, len(images_list)):
	os.rename(images_list[i],images_new_list[i])