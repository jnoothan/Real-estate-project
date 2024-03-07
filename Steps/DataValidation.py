# Step 1

import os
import pandas as pd
import shutil
from zenml import step


# segregate csv files into valid and invalid data folders by feature
@step
def Datavalidation(path):
    for i in os.listdir(path=path):
        if i.endswith(".csv"):
            current_csv_path = os.path.join(path, i)
            current_csv = pd.read_csv(current_csv_path)
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
                shutil.move(
                    os.path.join(current_csv_path), os.path.join("Valid_Data", i)
                )
            else:
                shutil.move(
                    os.path.join(current_csv_path), os.path.join("InvalidData", i)
                )

    print("csv files segregated. Check Ingested Data and InvalidData folders!")


if __name__=="__main__":

    folder_path = '.\Data for ml'
    Datavalidation(folder_path)