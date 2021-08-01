import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import OneHotEncoder


class Model:
    data = []
    model = None
    def __init__(self, data):
        self.data = data

    def convert_data(self):
        # drop unwanted dataframes (like redudant ones such as dates)
        # y = self.data['decile_score']

        self.data = self.data[
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

        # for column in self.data:
        #     print(self.data[column])
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print("Original number of columns:", len(self.data.columns))

        # print("Dropping out column \'id'\'")
        # self.data = self.data.drop(['id'], 1)

        # print("Dropping out column \'decile_score'\'")
        # self.data = self.data.drop(['decile_score'], 1)
        # # print(len(self.data.columns))

        #print("Picking out non numeric columns...")
        x = self.data.select_dtypes(include=[object])
        # print(x.columns)
        #print("Number of non numerical columns:", x.shape)

        # go through the columns of data
        cols = []
        for i in range(len(self.data.columns)):
            col = self.data.iloc[:, i]
            # if its a categorical column drop it from data (because we're going to add the 1 hot versions above)
            if col.dtype == "object":
                #print(col.name)
                cols.append(col.name)
        self.data = self.data.drop(columns=cols, axis=1)
        #print(self.data.shape)

        le = preprocessing.LabelEncoder()
        X_2 = x.apply(le.fit_transform)
        #print(X_2.shape)
        # print(X_2.head(2))
        enc = preprocessing.OneHotEncoder()
        enc.fit(X_2)
        onehotlabels = enc.transform(X_2).toarray()
        #print(type(onehotlabels))
        #print("Hot label dimensions:", onehotlabels.shape)

        labelsdf = pd.DataFrame(onehotlabels)

        # concatenate 1hot encoded dataframe (x) with original dataframe(that now has dropped the redudant columns)
        #print(type(self.data), type(labelsdf))
        # frames = [self.data, labelsdf]
        # result = pd.concat(frames)
        result = pd.concat([self.data, labelsdf], axis=1)
        #print("Shape:", result.shape)
        #print(result.head(2))

        result = result.dropna()
        y = pd.DataFrame(result, columns=["decile_score"])
        result = result.drop(["decile_score"], 1)
        #print(result.shape)

        # print(onehotlabels.shape)
        # print(onehotlabels)
        # Normal linear regression model

        # returns build_model's return values (X_test, y_test) for further testing
        return self.build_model(result, y)

    def build_model(self, X, y):
        print("Spliting dataset 80/20...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        print(X_train.shape, X_test.shape)
        print("Training...")
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        x_pred = self.model.predict(X_train)
        print("Model train score:", self.model.score(X_train, y_train))
        print("Model test score:", self.model.score(X_test, y_test))

        # The coefficients
        #print("Coefficients: \n", model.coef_)
        # The mean squared error
        print("MSE Train: %.2f" % mean_squared_error(y_train, x_pred))
        print("MSE Test: %.2f" % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
        print()

        return X_test, y_test
        # print("Plotting result...")
        # plt.scatter(X_test, y_test, color='black')
        # plt.plot(X_test, y_pred, color='blue', linewidth=3)
        #
        # plt.xticks(())
        # plt.yticks(())
        #
        # plt.show()
 
    # TODO: generalize this to be able to work with any attribute
    def score_attribute(self, sample, d_score): # should be same input shape as model e.g. X_test
        sample = numpy.asarray(sample)[0]
        print("compas score: " + str(d_score))

        race_scores = []
        # test 4 different race versions of each sample
        for i in range(4):
            # zero race 1-hot
            for j in range(10, 14, 1):
                sample[j] = 0
            sample[i] = 1
            race_scores.append(self.model.predict([sample])[0][0])
            print('race ' + str(i) + ' score: ' + str(race_scores[i]))
        print('---------------------------')

        return race_scores 




