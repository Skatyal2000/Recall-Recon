# Import the libraries
import requests
import csv
import time
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv("keys.env")


BASE_URL = os.getenv("API_Key") # API url
LIMIT = 1000  
OUTPUT_FILE = "vehicle_recalls.csv"  # Ouput dataset file name intialization

def fetch_recalls():  # Function to fetch the data from API 
    all_data = []
    offset = 0

    while True:
        url = f"{BASE_URL}?$limit={LIMIT}&$offset={offset}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if not data:
                break  # Stop if no more data is returned

            all_data.extend(data) # Add all json responses to list
            offset += LIMIT  # Move to the next batch
            time.sleep(1)  # Avoid rate limiting
            
            """ To ensure we retrieve the complete data without losing any information, we limit each fetch to 1000 calls and introduce a 1-second                   delay between fetches. This approach helps maintain data integrity."""

        except requests.exceptions.RequestException as e:
            print(f"Error fetching recall data: {e}")
            break

    return all_data   # Returning all the fetched data from the API in JSON format as a list.

def save_to_csv(data, filename): # Function to transform JSON data into a human-readable CSV format.
    if not data:
        print("No data to save.")
        return

    all_keys = set()
    for entry in data:
        all_keys.update(entry.keys())

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_keys))
        writer.writeheader()
        for entry in data:
            writer.writerow({key: entry.get(key, '') for key in all_keys})


if __name__ == "__main__":
    recalls = fetch_recalls()
    save_to_csv(recalls, OUTPUT_FILE)
    print(f"Recall data saved to {OUTPUT_FILE}")
