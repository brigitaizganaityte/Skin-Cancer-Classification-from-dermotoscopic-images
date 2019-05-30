import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt


#reading all csv files from directory 
all_files = glob.glob("C:/Users/Brigita/Desktop/Bakalaurinis/findEdgesResult/" 
                      + "/*.csv")

data = []
#creating array from 0 to 599 for pixels representation
six = np.arange(600)
#making 1D numpy array
six = np.reshape(six, (600, 1))	
#creating array from 0 to 599 for pixels representation
four_fifty = np.arange(450)
#making 1D numpy array
four_fifty = np.reshape(four_fifty, (450, 1))	

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df.columns = ['X', 'Y', 'Value', 'Label'] 
    full_df_sorted_by_x = df.sort_values(['X', 'Y'], ascending=[True, True])
    
    no_dub_1 = full_df_sorted_by_x.drop_duplicates(subset='X', keep="first")
    no_dub_2 = df.drop_duplicates(subset='Y', keep="first")
    no_dub_3 = full_df_sorted_by_x.drop_duplicates(subset='X', keep="last")
    no_dub_4 = df.drop_duplicates(subset='Y', keep="last")
    id = (filename[77:])    
	#creating arrays for each picture with data:
    #600 of X values of first projection, 600 of Y values of first projection
    #450 of X values of second projection, 450 of Y values of second projection
    #600 of X values of third projection, 600 of Y values of third projection
    #450 of X values of fourth projection, 450 of Y values of fourth projection
	####first projection
    y_data_1 = no_dub_1.loc[:, ['Y']].values
    x_data_1 = no_dub_1.loc[:, ['X']].values
    starty = np.repeat(450,x_data_1[0])
    endy = np.repeat(450, 600-(len(y_data_1)+len(starty)))
    startyr = np.reshape(starty, (len(starty), 1))	
    endyr = np.reshape(endy, (len(endy), 1))
    result1 = np.concatenate((six, startyr, y_data_1, endyr))
	####second projection
    x_data_2 = no_dub_2.loc[:, ['X']].values
    y_data_2 = no_dub_2.loc[:, ['Y']].values
    startx = np.repeat(600,y_data_2[0])
    endx = np.repeat(600, 450-(len(x_data_2)+len(startx)))
    startxr = np.reshape(startx, (len(startx), 1))	
    endxr = np.reshape(endx, (len(endx), 1))
    result2 = np.concatenate((startxr, x_data_2, endxr, four_fifty))
    ####third projection
    x_data_3 = no_dub_3.loc[:, ['X']].values
    y_data_3 = no_dub_3.loc[:, ['Y']].values
    starty = np.repeat(450,x_data_3[0])
    endy = np.repeat(450, 600-(len(y_data_3)+len(starty)))
    startyr = np.reshape(starty, (len(starty), 1))	
    endyr = np.reshape(endy, (len(endy), 1))
    result3 = np.concatenate((six, startyr, y_data_3, endyr))
    ###fourth projection
    x_data_4 = no_dub_4.loc[:, ['X']].values
    y_data_4 = no_dub_4.loc[:, ['Y']].values
    startx = np.repeat(600,y_data_4[0])
    endx = np.repeat(600, 450-(len(x_data_4)+len(startx)))
    startxr = np.reshape(startx, (len(startx), 1))	
    endxr = np.reshape(endx, (len(endx), 1))
    result4 = np.concatenate((startxr, x_data_4, endxr, four_fifty))
	#concatenating data to one array
    final_image = np.concatenate((result1, result2, result3, result4))
    
    #creating the final csv file
    with open("C:/Users/Brigita/Desktop/Bakalaurinis/res_pca.csv", 'a') as f:
       pd.DataFrame(final_image).to_csv(f, header=False, index=None)


