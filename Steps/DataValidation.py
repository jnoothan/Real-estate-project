import os
import pandas as pd
import click
import shutil

# segregate csv files into valid and invalid data folders by features
@click.command()
@click.option('--path', help='Folder Path of the batch data')

def DataValidation(path):
    for i in os.listdir(path=path):
        if i.endswith('.csv'):
            current_csv_path = os.path.join(path,i)
            current_csv = pd.read_csv(current_csv_path)
            std_columns=['status', 'bed', 'bath', 'acre_lot', 'city', 'state', 'house_size','price']
            if set(current_csv.columns)==set(std_columns):
                shutil.move(os.path.join(current_csv_path), os.path.join('ValidData',i))
            else:
                shutil.move(os.path.join(current_csv_path), os.path.join('InvalidData',i))
    
    print('csv files segregated. Check ValidData and InvalidData folders!')
    


if __name__=='__main__':
    DataValidation()