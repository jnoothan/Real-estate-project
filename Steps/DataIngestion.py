import os
import pandas as pd
import click
import shutil

# Specify folder path to batch data and merge the data

#@click.command()
#@click.option('--path', help='Folder path')
relative_path='ValidData'

def data_ingestion(relative_path):
    df=pd.DataFrame()
    for csv_file in os.listdir(path=relative_path):
        current_csv=os.path.join(relative_path,csv_file)
        current_df=pd.read_csv(current_csv)
        df=pd.concat([df,current_df],ignore_index=True)
    
    # Merge all batch Data
    merged_csv_name='Data_merged_for_ml_training.csv'
    merged_csv_path=os.path.join(relative_path,merged_csv_name)
    df.to_csv(merged_csv_path,index=False)
    
    # Move the merged 
    Destination_folder = 'Ingested Data'
    Existing_csv_path = os.path.join(relative_path,merged_csv_name)
    Destination_csv_path = os.path.join(Destination_folder,merged_csv_name)
    shutil.move(Existing_csv_path,Destination_csv_path)
    
    for csv_file in os.listdir(path=relative_path):
        os.remove(csv_file)
    
    print('Data Ingested')
    return df

        
        

if __name__=='__main__':
    data_ingestion(relative_path)