import csv
import os

# Read all the CSV files in the directory
csv_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv')]

# Open the new CSV file
with open('future_temp.csv', 'w', newline='') as result_file:
    w = csv.writer(result_file)

    # Loop through each CSV file and merge them into the new CSV file
    for f in csv_files:
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                w.writerow(row)