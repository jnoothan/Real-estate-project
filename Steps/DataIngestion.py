import os
import pandas as pd
import click

# specify folder path to batch data and merge the data

# Specify folder path to batch data and merge the data
@click.command()
@click.option('--path', help='Folder path')

def data_ingestion(path):
    df=pd.DataFrame()
    for i in os.listdir(path=path):
        if i.endswith('.csv'):
            csv_file=os.path.join(path,i)
            current_df=pd.read_csv(csv_file)
            df=pd.concat([df,current_df],ignore_index=True)
    
    print(df)
    return df

        
        

if __name__=='__main__':
    data_ingestion()