import os
import tkinter

import pandas as pd  # For reading CSV files
import time
from plotgen import Window
from LRModel import Model

# Path to all CSVs
DATA_PATH = 'csv_data'


def read_records():
    all_records = []
    length = 0

    # Array containing a list of CSV files:
    record_data = os.listdir(DATA_PATH)

    for record in record_data:
        print("Reading " + record)
        data = pd.read_csv(DATA_PATH + "/" + record)
        length += data.shape[0]
        all_records.append(data)

    # Create the dataframe and return it.
    all_records = pd.concat(all_records, axis=0, ignore_index=True)
    print("Total number of rows: {}".format(length))
    return all_records


if __name__ == "__main__":
    start = time.time()
    # root = tkinter.Tk()
    # window = Window(root, read_records())
    # tkinter.mainloop()
    model = Model(read_records())
    model.convert_data()
    elapsed = time.time() - start
    print("\n\nScript execution time: {} seconds".format(elapsed))
