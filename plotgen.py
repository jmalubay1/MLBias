import tkinter
import os
import pandas as pd
import matplotlib.pyplot as plt
from LRModel import *
from matplotlib.ticker import IndexLocator, AutoLocator

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Path to all CSVs
DATA_PATH = "csv_data"


class Window:
    def __init__(self, root, records):
        self.records = self.convert_data(records)
        print(self.records)
        self.root = root
        self.root.wm_title("Figure Generator")

        self.var1_select = tkinter.StringVar(self.root)
        self.var1_select.set("X Variable")
        self.v1 = tkinter.OptionMenu(
            self.root, self.var1_select, *self.records.columns.values
        )
        self.v1.grid(row=0, column=0)

        self.var2_select = tkinter.StringVar(self.root)
        self.var2_select.set("Y Variable")
        self.v2 = tkinter.OptionMenu(
            self.root, self.var2_select, *self.records.columns.values
        )
        self.v2.grid(row=0, column=1)

        self.gen_button = tkinter.Button(
            master=root, text="Generate", command=self._fig_gen
        )
        self.gen_button.grid(row=0, column=2)

        self.fig = plt.Figure()
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
            print(count)
            self.fig.clf()
            scale = 5
        
            ax = self.fig.add_subplot(111)
            
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.scatter(x=count[x_var],y=count[y_var],s=count['total']*scale, c=count['total'])
            ax.margins(x=None,y=1)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.draw()

            self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

    def convert_data(self,records):

        data = records[
            [
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
                "decile_score",
            ]
        ]

        #data = data.dropna()
        return data


def read_records():
    all_records = []
    length = 0

    # Array containing a list of CSV files:
    record_data = os.listdir(DATA_PATH)

    for record in record_data:
        #print("Reading " + record)
        data = pd.read_csv(DATA_PATH + "/" + record)
        length += data.shape[0]
        all_records.append(data)

    # Create the dataframe and return it.
    all_records = pd.concat(all_records, axis=0, ignore_index=True)
    #print("Total number of rows: {}".format(length))
    #print(all_records)
    return all_records


if __name__ == "__main__":
    recs = read_records()
    root = tkinter.Tk()
    my_win = Window(root, recs)
    tkinter.mainloop()
