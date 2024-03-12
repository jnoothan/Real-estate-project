import os
import pandas as pd
from zenml import step
import shutil
import logging
from typing import Tuple, Annotated
# Set up logging
logging.basicConfig(filename='data_ingestion.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@step
def data_ingestion(relative_path:str)-> Annotated[pd.DataFrame,'Ingested_Data']:
    # Initialize an empty list to store DataFrames
    df_list = []

    for csv_file in os.listdir(relative_path):
        current_csv_path = os.path.join(relative_path, csv_file)
        try:
            current_df = pd.read_csv(current_csv_path)
            df_list.append(current_df)  # Append the DataFrame to the list
        except pd.errors.EmptyDataError:
            logging.error(f"Error: File {csv_file} has no columns.")
            print(f"Error: File {csv_file} has no columns.")

    # Concatenate all DataFrames in the list
    try:
        df = pd.concat(df_list, ignore_index=True)
    except ValueError as e:
        logging.error(f"Error concatenating DataFrames: {e}")
        print(f"Error concatenating DataFrames: {e}")
        return None

    if df.empty:
        logging.error("Error: No data found in any CSV file.")
        print("Error: No data found in any CSV file.")
        return None

    merged_csv_name = "Ingested_Data.csv"
    merged_csv_path = os.path.join(relative_path, merged_csv_name)
    df.to_csv(merged_csv_path, index=False)

    Destination_folder = "Ingested Data"
    Destination_csv_path = os.path.join(Destination_folder, merged_csv_name)
    try:
        shutil.move(merged_csv_path, Destination_csv_path)
    except Exception as e:
        logging.error(f"Error moving file: {e}")
        print(f"Error moving file: {e}")

    for csv_file in os.listdir(relative_path):
        try:
            os.remove(os.path.join(relative_path, csv_file))
        except Exception as e:
            logging.error(f"Error removing file: {e}")
            print(f"Error removing file: {e}")

    print("Data Ingested")
    df.to_csv('./Ingested Data/Ingested_data.csv', index=False)
    print(type(df))
    return df


if __name__ == '__main__':
    relative_path = "./Valid_Data"
    data_ingestion(relative_path)
