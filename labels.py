import glob
import numpy as np
import pandas as pd

#reading all csv files from directory 
all_files = glob.glob("C:/Users/Brigita/Desktop/Bakalaurinis/findEdgesResult/" 
                      + "/*.csv")

label = []
#appending right label while subtracting it from each filename 
for filename in all_files:
   diagnoses = filename.split('_')  
   if (diagnoses[-2] == 'nv'):
	    label.append(3)
   elif (diagnoses[-2] == 'mel'):
	    label.append(4)
   elif (diagnoses[-2] == 'bkl'):
	    label.append(1)
   elif (diagnoses[-2] == 'akiec'):
	    label.append(0)
   elif (diagnoses[-2] == 'df'):
	    label.append(2)
   elif (diagnoses[-2] == 'bcc'):
	    label.append(5)
   else:
        label.append(6)
#creating separate csv file for the labels of given data points
#appending information
with open("C:/Users/Brigita/Desktop/Bakalaurinis/labels.csv", 'a') as f:        
	    pd.DataFrame(label).to_csv(f, header=False, index=None)

	
