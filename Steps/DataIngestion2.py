import os
import pandas as pd
from zenml import step
import shutil
import logging
import mysql.connector
from mysql.connector import Error
from typing import Tuple, Annotated

# Set up logging
logging.basicConfig(filename='data_ingestion.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# MySQL connection setup
def create_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            port=int(os.getenv('MYSQL_PORT')), 
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE')
        )
        return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        print(f"Error connecting to MySQL: {e}")
        return None

# Log each step into MySQL
def log_step(step_name: str, status: str, message: str = None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    step_name VARCHAR(255),
                    status VARCHAR(50),
                    message TEXT,
                    log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                INSERT INTO ingestion_logs (step_name, status, message)
                VALUES (%s, %s, %s)
            """, (step_name, status, message))
            connection.commit()
        except Error as e:
            logging.error(f"Error logging step: {e}")
            print(f"Error logging step: {e}")
        finally:
            cursor.close()
            connection.close()

@step
def data_ingestion(relative_path:str) -> Annotated[pd.DataFrame, 'Ingested_Data']:
    # Initialize an empty list to store DataFrames
    df_list = []
    
    # Log step: Starting ingestion
    log_step("Ingestion Start", "In Progress", f"Started ingesting from {relative_path}")
    
    for csv_file in os.listdir(relative_path):
        current_csv_path = os.path.join(relative_path, csv_file)
        try:
            current_df = pd.read_csv(current_csv_path)
            df_list.append(current_df)  # Append the DataFrame to the list
            log_step("Read CSV", "Success", f"Successfully read {csv_file}")
        except pd.errors.EmptyDataError:
            error_message = f"Error: File {csv_file} has no columns."
            logging.error(error_message)
            log_step("Read CSV", "Failed", error_message)
            print(error_message)

    # Concatenate all DataFrames in the list
    try:
        df = pd.concat(df_list, ignore_index=True)
        log_step("Concatenate DataFrames", "Success", "Successfully concatenated DataFrames")
    except ValueError as e:
        error_message = f"Error concatenating DataFrames: {e}"
        logging.error(error_message)
        log_step("Concatenate DataFrames", "Failed", error_message)
        print(error_message)
        return None

    if df.empty:
        error_message = "Error: No data found in any CSV file."
        logging.error(error_message)
        log_step("Check DataFrame", "Failed", error_message)
        print(error_message)
        return None

    merged_csv_name = "Ingested_Data.csv"
    merged_csv_path = os.path.join(relative_path, merged_csv_name)
    
    # Save the merged CSV
    try:
        df.to_csv(merged_csv_path, index=False)
        log_step("Save CSV", "Success", f"Saved CSV to {merged_csv_path}")
    except Exception as e:
        error_message = f"Error saving CSV: {e}"
        logging.error(error_message)
        log_step("Save CSV", "Failed", error_message)
        print(error_message)
    
    Destination_folder = "Ingested Data"
    Destination_csv_path = os.path.join(Destination_folder, merged_csv_name)
    
    # Move the merged CSV
    try:
        shutil.move(merged_csv_path, Destination_csv_path)
        log_step("Move CSV", "Success", f"Moved CSV to {Destination_csv_path}")
    except Exception as e:
        error_message = f"Error moving file: {e}"
        logging.error(error_message)
        log_step("Move CSV", "Failed", error_message)
        print(error_message)

    # Remove original CSV files
    for csv_file in os.listdir(relative_path):
        try:
            os.remove(os.path.join(relative_path, csv_file))
            log_step("Remove CSV", "Success", f"Removed {csv_file}")
        except Exception as e:
            error_message = f"Error removing file: {e}"
            logging.error(error_message)
            log_step("Remove CSV", "Failed", error_message)
            print(error_message)

    log_step("Ingestion End", "Success", "Data ingestion completed successfully")
    print("Data Ingested")
    
    df.to_csv('./Ingested Data/Ingested_data.csv', index=False)
    return df

if __name__ == '__main__':
    relative_path = "./Valid_Data"
    data_ingestion(relative_path)
