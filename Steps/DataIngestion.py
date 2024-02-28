import os
import pandas as pd
from zenml import step
import shutil


@step
def data_ingestion(relative_path):
    # Initialize an empty list to store DataFrames
    df_list = []

    for csv_file in os.listdir(relative_path):
        current_csv_path = os.path.join(relative_path, csv_file)
        try:
            current_df = pd.read_csv(current_csv_path)
            df_list.append(current_df)  # Append the DataFrame to the list
        except pd.errors.EmptyDataError:
            print(f"Error: File {csv_file} has no columns.")

    # Concatenate all DataFrames in the list
    df = pd.concat(df_list, ignore_index=True)

    if df.empty:
        print("Error: No data found in any CSV file.")
        return None

    merged_csv_name = "Data_merged_for_ml_training.csv"
    merged_csv_path = os.path.join(relative_path, merged_csv_name)
    df.to_csv(merged_csv_path, index=False)

    Destination_folder = "Ingested Data"
    Destination_csv_path = os.path.join(Destination_folder, merged_csv_name)
    shutil.move(merged_csv_path, Destination_csv_path)

    for csv_file in os.listdir(relative_path):
        os.remove(os.path.join(relative_path, csv_file))

    print("Data Ingested")
    df.to_csv('./Ingested Data/Ingested_data.csv')
    return df


if __name__ == '__main__':
    relative_path = "./Ingested Data"
    data_ingestion(relative_path)
