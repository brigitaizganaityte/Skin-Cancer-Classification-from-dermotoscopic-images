import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

#reading all csv files from directory 
all_files = glob.glob("C:/Users/Brigita/Desktop/Bakalaurinis/findEdgesData/" 
                       + "/*.csv")
data = []
for filename in all_files:
	full_df = pd.read_csv(filename, index_col=None, header=0)
	#dropping dataframe rows which pixel value < 253 because 
	#it is not representing black color
	without_zero = full_df[full_df.Value > 253]	
	#getting label of each filename
	filename = filename[:-8]
	id = (filename[74:])
	label = filename.split('_')		
	without_zero['Label']=label[-1]
    #creating csv files with data points with values > 253
	with open('C:/Users/Brigita/Desktop/Bakalaurinis/findEdgesResult/'
	          +id+'_0.csv', 'w') as f:
		without_zero.to_csv(f, header=False, index=None)
