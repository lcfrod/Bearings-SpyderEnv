# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:53:52 2022

@author: Luiz Rodrigues

Program : FEMTO_Merge_Dataset
Purpose : Given a /folder/file_name, from the  FEMTO Bearing dataset, merge all the csv
          files of a bearing in one file. 
 
History 
-------
Date          Author    Description
2021-09-05 -  LCR     - Creation
2022-01-30 -  LCR     - Modified / MTP
"""

import glob
import numpy  as np
import pandas as pd

print( "FEMTO_Merge_Dataset - Starting the Process.")

Fs     = 25600      # Sample Frequency  26,6 KHz
period = 1 / Fs     # Sampling Interval   

def csvs_merge(path_in, path_out, file_name, types_infos):
    nf = 0
    for file_type, type_infos in types_infos.items():
       
        # Gathering all .csv files from type  'file_type'
        csv_files = sorted(glob.glob('%s/%s*.csv' % (path_in, file_type)))
        
        # Get the number of files in the folder
        nf = len(csv_files) 
        
        # Some folders dont have all the file types
        if not csv_files:
            continue
            
        # Determining the separator type with the first  .csv file
        reader = pd.read_csv(csv_files[0], sep=None, iterator=True, engine='python')
        inferred_sep = reader._engine.data.dialect.delimiter
        
        # Merging  .csv files of type 'file_type'
        merged_csv = pd.concat([ pd.read_csv('%s' % (f) ,
                                sep = inferred_sep,
                                usecols = type_infos['usecols'] ,
                                names = type_infos['names'],
                                header=None, engine='c') for f in csv_files ] )
                                          

        merged_csv.to_csv("%s/%s.csv" % (path_out, file_name), index=False, encoding='utf-8-sig' )
       
   
    return ( path_out +'/'+file_name +'.csv', nf )



# Name of the bearing files to merge
file_name = 'Bearing1_3'


# Inform the Folder of records to merge
input_original_dataset_path = 'C:/FEMTO/Full_Test_Set/'+file_name

# Inform the Folder to store the merged file
output_merged_data_path = 'C:/FEMTO/Dataset'


types_infos = {
    # file type identifier, columns to read, columns name ( Considering only acceleration)
    'acc' : {'usecols' : [0,1,2,3,4,5], 'names' : ['hour', 'min', 'sec', 'usec', 'acc_horz', 'acc_vert']}
}


print("Please, wait. Creating the file  {}.".format(file_name))

# Scan and Merge all the files in the folder
merged_file, nf = csvs_merge(input_original_dataset_path, 
                             output_merged_data_path, 
                             file_name, 
                             types_infos)

# After merged, read the file for format timestamp
df = pd.read_csv(merged_file)
merged_len = len(df) 

# (Experimenting )Compose the String DateTime from Hour Min and Sec + Microsseconds into Microsseconds format  

# Assume the initial date of the epoch
#w_date = '1970-01-01 '

# Compose the Datetime from Strings and stores it in a Datetime format
#df['dt_time'] = pd.to_datetime( w_date + df['hour'].astype(str) + ':'+ df['min'].astype(str) + ':'+ df['sec'].astype(str))

# Convert Datetime to Timestamp to Nanossecods and then convert to Microsseconds (* 1000).  Store as Integer
#df['dt_usec'] = (df['dt_time'].values.astype(np.int64) / 1000).astype(np.int64)


# Now format the residual Microsseconds as Integer
#df['ts'] =  df['dt_usec'] + df['usec'].values.astype(np.int64)  

# Create Decimal timestamp in Seconds
df['ts'] =  np.linspace(0.0000, period * len(df), len(df))
df['ts'] = df['ts'].apply(lambda x: '%.6f' % x)

#  Clean up and reorder the columns
df.drop(['hour', 'min', 'sec', 'usec'], axis=1, inplace=True)
df = df[['ts', 'acc_horz', 'acc_vert']]    

# Save the merged file, without index col
df.to_csv( merged_file, index = False )


print("Done. {} was created from {} files.".format(merged_file, nf))

print( "Ending the Process.")




