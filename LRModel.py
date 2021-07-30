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

    def __init__(self, data):
        self.data = data

    def convert_data(self):
        # for column in self.data:
        #     print(self.data[column])
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
        print("Original number of columns:", len(self.data.columns))

        print("Dropping out column \'id'\'")
        self.data = self.data.drop(['id'], 1)

        y = self.data['decile_score']

        print("Dropping out column \'decile_score'\'")
        self.data = self.data.drop(['decile_score'], 1)
        # print(len(self.data.columns))

        print("Picking out non numeric columns...")
        x = self.data.select_dtypes(include=[object])
        # print(x.columns)
        print("Number of non numerical columns:", x.shape)

        le = preprocessing.LabelEncoder()
        X_2 = x.apply(le.fit_transform)
        # print(X_2.head(2))
        enc = preprocessing.OneHotEncoder()
        enc.fit(X_2)
        onehotlabels = enc.transform(X_2).toarray()
        print("Hot label dimensions:", onehotlabels.shape)
        # print(onehotlabels.shape)
        # print(onehotlabels)
        # Normal linear regression model
        self.build_model(onehotlabels, y)

    def build_model(self, X, y):
        print("Spliting dataset 80/20...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(X_test.shape, y_test.shape)
        print("Training...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # print(y_pred)
        print("Model score:", model.score(X_test, y_test))
        print(model.score(X_test, y_test))

        # The coefficients
        print('Coefficients: \n', model.coef_)
        # The mean squared error
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

        # print("Plotting result...")
        # plt.scatter(X_test, y_test, color='black')
        # plt.plot(X_test, y_pred, color='blue', linewidth=3)
        #
        # plt.xticks(())
        # plt.yticks(())
        #
        # plt.show()

