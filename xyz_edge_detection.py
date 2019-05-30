import pandas as pd
import numpy as np

#header none - reads the first line of the file, too.
data_df = pd.read_csv('C:/Users/Brigita/Desktop/Bakalaurinis/full_res_pca.csv', 
                      header = None) 
data_df.columns = ['labels','component1', 'component2', 'component3']

#number of rows for the xyz file
rows = data_df.labels.count() 		
#writing dataframe to xyz file
file = open('C:/Users/Brigita/Desktop/Bakalaurinis/Hist_edge.xyz','a')
#first row of the file - number of records 
file.write(str(rows) + '\n')
#second row of the file - title
file.write("Edge detection data" + '\n')
#writing results
np.savetxt(file, data_df.values, fmt='%s')
file.close()