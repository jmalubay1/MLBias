import os
import tkinter
import numpy as np
import random as r

import pandas as pd  # For reading CSV files
import time

from pandas.core.frame import DataFrame
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


def test_data(num_records):
    races = ['A', 'B', 'C', 'D']
    sexs = ['F','M']

    dtypes = np.dtype(
        [
            ("sex", str),
            ("age", int),
            ("race", str),
            ("juv_fel_count", int),
            ("juv_misd_count", int),
            ("juv_other_count", int),
            ("priors_count", int),
            ("days_b_screening_arrest", int),
            ("c_days_from_compas", int),
            ("c_charge_degree", str),
            ("is_recid", int),
            ("r_charge_degree", str),
            ("decile_score", int)
        ]
    )

    data = np.empty(0,dtype=dtypes)
    testframe = DataFrame(data)
    
    for i in range(num_records):
        race = r.choice(races)
        age = r.randint(18,80)
        sex = r.choice(sexs)
        record = rand_record(sex,age,race,testframe.columns)
        testframe = testframe.append(record, ignore_index=True)

    #print(testframe.shape)
    return testframe

def rand_record(sex,age,race,columns):
    record = [sex,age,race]

    #juv_fel_count
    record.append(int(r.randint(0,3)))
    #juv_misd_count
    record.append(r.randint(0,3))
    #juv_other_count
    record.append(r.randint(0,3))
    #priors_count
    record.append(r.randint(0,5))
    #days_b_screening_arrest
    record.append(r.randint(0,10))
    #c_days_from_compas
    record.append(r.randint(0,4))
    #c_charge_degree
    record.append(r.randint(0,3))
    #is_recid
    record.append(r.randint(0,1))
    #r_charge_degree
    if record[10] == 1:
        record.append(r.choice(['F','M']))
    else:
        record.append('')
    #decile_score
    score = 1
    if race == 'B':
        score += 2
    if race == 'C':
        score += 5
    if race == 'D':
        score += 8
    record.append(score)

    rec_dict = {columns[i]: record[i] for i in range(len(record))}
    
    return rec_dict


if __name__ == "__main__":
    start = time.time()
    # root = tkinter.Tk()
    # window = Window(root, read_records())
    # tkinter.mainloop()

    # Random Dataframe
    # rand_model = Model(test_data(1000))
    # rand_model.convert_data()

    print('----------------')

    compas_model = Model(read_records())
    X_test, y_test = compas_model.convert_data()

    # control for attribute (currently hardcoded to race) and compare scores
    for i in range(10):
        compas_model.score_attribute(X_test.iloc[[i]], y_test.iat[i, 0]) # input shape is a single df row and the recidivism score

    elapsed = time.time() - start
    print("\n\nScript execution time: {} seconds".format(elapsed))
