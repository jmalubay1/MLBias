import tkinter
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Path to all CSVs
DATA_PATH = "csv_data"


class Window:
    def __init__(self, root, records):
        self.records = records
        self.root = root
        self.root.wm_title("Figure Generator")

        self.var1_select = tkinter.StringVar(self.root)
        self.var1_select.set("X Variable")
        self.v1 = tkinter.OptionMenu(
            self.root, self.var1_select, *records.columns.values
        )
        self.v1.grid(row=0, column=0)

        self.var2_select = tkinter.StringVar(self.root)
        self.var2_select.set("Y Variable")
        self.v2 = tkinter.OptionMenu(
            self.root, self.var2_select, *records.columns.values
        )
        self.v2.grid(row=0, column=1)

        self.gen_button = tkinter.Button(
            master=root, text="Generate", command=self._fig_gen
        )
        self.gen_button.grid(row=0, column=2)

        self.fig = Figure(figsize=(15, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

        self.quit_button = tkinter.Button(
            master=self.root, text="Quit", command=self._quit
        )
        self.quit_button.grid(row=2, column=0, columnspan=3)

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def _fig_gen(self):
        x_var, y_var = self.var1_select.get(), self.var2_select.get()
        if x_var in self.records.columns and y_var in self.records.columns:
            count = self.records.groupby([x_var,y_var]).size()
            count = pd.DataFrame(count.reset_index())
            count.columns = [x_var,y_var,'total']
            count = count.pivot(index=y_var, columns=x_var, values='total')
            count = count.iloc[::-1]
            print(count)
            self.fig.clf()
            sns.heatmap(count, cmap='Oranges', annot=True, fmt='g', ax=self.fig.subplots())
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.draw()

            self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)


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
    print(all_records)
    return all_records


if __name__ == "__main__":
    recs = read_records()
    root = tkinter.Tk()
    my_win = Window(root, recs)
    tkinter.mainloop()
