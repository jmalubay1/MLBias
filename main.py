import os
import tkinter
import numpy as np

import pandas as pd  # For reading CSV files
import time
from plotgen import Window
from LRModel import Model

# Path to all CSVs
DATA_PATH = "csv_data"


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

    df = pd.DataFrame(
        np.random.randint(0, 100, size=(1000, 12)),
        columns={
            "sex",
            "age",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "days_b_screening_arrest",
            "c_days_from_compas",
            "c_charge_degree",
            "is_recid",
            "r_charge_degree",
        },
    )
    y = pd.DataFrame(np.random.randint(0, 10, size=(1000, 1)), columns={"decile_score"})
    model = Model(df)
    model.build_model(df, y)

    # model = Model(read_records())
    # model.convert_data()
    elapsed = time.time() - start
    print("\n\nScript execution time: {} seconds".format(elapsed))
