import os
import pandas as pd
import shutil
import logging
from zenml import step

# Set up logging
logging.basicConfig(filename='data_validation.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@step(enable_cache=False)
def DataValidation(path:str)->None:
    try:
        for i in os.listdir(path=path):
            if i.endswith(".csv"):
                current_csv_path = os.path.join(path, i)
                try:
                    current_csv = pd.read_csv(current_csv_path)
                except pd.errors.ParserError as e:
                    logging.error(f"Error parsing CSV file '{i}': {e}")
                    print(f"Error parsing CSV file '{i}': {e}")
                    continue

                std_columns = [
                    "status",
                    "bed",
                    "bath",
                    "acre_lot",
                    "city",
                    "state",
                    "house_size",
                    "price",
                ]
                if set(current_csv.columns) == set(std_columns):
                    try:
                        shutil.move(
                            os.path.join(current_csv_path), os.path.join("Valid_Data", i)
                        )
                    except Exception as e:
                        logging.error(f"Error moving file '{i}' to Valid_Data: {e}")
                        print(f"Error moving file '{i}' to Valid_Data: {e}")
                else:
                    try:
                        shutil.move(
                            os.path.join(current_csv_path), os.path.join("Invalid_Data", i)
                        )
                    except Exception as e:
                        logging.error(f"Error moving file '{i}' to Invalid_Data: {e}")
                        print(f"Error moving file '{i}' to Invalid_Data: {e}")
    except Exception as e:
        logging.error(f"Error processing files: {e}")
        print(f"Error processing files: {e}")

    print("CSV files segregated. Check Valid_Data and Invalid_Data folders!")


if __name__ == "__main__":
    folder_path = './Data for ml'
    DataValidation(folder_path)
