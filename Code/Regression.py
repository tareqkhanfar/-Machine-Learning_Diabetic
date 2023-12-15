import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import  matplotlib.pyplot as plt
import seaborn as sns
from DataReader import ReadDataSet

class RegressionTask:
    @staticmethod
    def inverseFromScalledToOrginal( min_original  , max_original , scaled_value):
        return  scaled_value * (max_original - min_original) + min_original

    @staticmethod
    def LR1 () :
        data = ReadDataSet()
        dataset = data.dataset
        print(dataset.head())
        x = dataset.drop('AGE' , axis=1)
        y = dataset['AGE']

        scaler = MinMaxScaler()
        X_scaled_data = scaler.fit_transform(x)
        print("X_scaled_data", X_scaled_data[:5])

        x_train , x_test , y_train , y_test = train_test_split(X_scaled_data , y , test_size=0.2 , random_state=0, shuffle=True )
        lr1 = LinearRegression()
        lr1.fit(x_train , y_train)
        ages_predicted = lr1.predict(x_test)
        MSE = mean_squared_error(ages_predicted , y_test)

       # correlationMatrix = dataset.corr()
        #sns.heatmap(correlationMatrix ,annot=True , linewidths=.5 )
        #plt.show()

        print("Y_TEST values : " , y_test[:10])
        print("Y hat : " , ages_predicted[:10])

        print("Parameters : " , lr1.coef_)
        print("Intercept : " , lr1.intercept_)
        print("MSE : " , MSE)
        RMSE = numpy.sqrt(MSE)
        print("RMSE : " , RMSE)
        correlation_matrix = dataset.corr()
        print("correlation matrix:\n", correlation_matrix)
        sns.heatmap(correlation_matrix, annot=True, linewidths=.5, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(y_test.reset_index(drop=True), label='True Values', color='green')
        plt.plot(ages_predicted, label='Predicted Values', color='red')
        plt.title('Comparison of True and Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Age')
        plt.legend()
        plt.show()

        differences = y_test - ages_predicted

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(differences)), differences, color='purple')
        plt.title('Difference Between True and Predicted Values')
        plt.xlabel('Data Point Index')
        plt.ylabel('Difference in Age')
        plt.show()

    @staticmethod
    def LR2() :
        data = ReadDataSet()
        dataset = data.dataset
        x = dataset['NPG'].values.reshape(-1, 1)
        print(x[:5])
        y = dataset['AGE']


        x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0 , shuffle=True )
        lr1 = LinearRegression()
        lr1.fit(x_train , y_train)
        ages_predicted = lr1.predict(x_test)
        MSE = mean_squared_error(ages_predicted , y_test)

       # correlationMatrix = dataset.corr()
        #sns.heatmap(correlationMatrix ,annot=True , linewidths=.5 )
        #plt.show()

        print("Y_TEST values : " , y_test[:10])
        print("Y hat : " , ages_predicted[:10])

        print("Parameters : " , lr1.coef_)
        print("Intercept : " , lr1.intercept_)
        print("MSE : " , MSE)
        RMSE = numpy.sqrt(MSE)
        print("RMSE : " , RMSE)
        correlation_matrix = dataset.corr()
        print("correlation matrix:\n", correlation_matrix)
        sns.heatmap(correlation_matrix, annot=True, linewidths=.5, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

        plt.scatter(x_test, y_test, color='blue')
        plt.plot(x_test, ages_predicted, color='red', linewidth=2)
        plt.title('Linear Regression (LR2)')
        plt.xlabel('NPG')
        plt.ylabel('AGE')
        plt.show()


        plt.figure(figsize=(10, 6))
        plt.plot(y_test.reset_index(drop=True), label='True Values', color='green')
        plt.plot(ages_predicted, label='Predicted Values', color='red')
        plt.title('Comparison of True and Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Age')
        plt.legend()
        plt.show()

        differences = y_test - ages_predicted

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(differences)), differences, color='purple')
        plt.title('Difference Between True and Predicted Values')
        plt.xlabel('Data Point Index')
        plt.ylabel('Difference in Age')
        plt.show()



    @staticmethod
    def LR3 () :
        data = ReadDataSet()
        dataset = data.dataset
        print(dataset.head())
        x = dataset[['NPG' , 'PGL' , 'DIA']]
        y = dataset['AGE']


        scaler = MinMaxScaler()
        X_scaled_data = scaler.fit_transform(x)
        print("X_scaled_data", X_scaled_data[:5])

        x_train , x_test , y_train , y_test = train_test_split(X_scaled_data , y , test_size=0.2 , random_state=0 , shuffle=True )
        lr3 = LinearRegression()
        lr3.fit(x_train , y_train)
        ages_predicted = lr3.predict(x_test)
        MSE = mean_squared_error(ages_predicted , y_test)




       # correlationMatrix = dataset.corr()
        #sns.heatmap(correlationMatrix ,annot=True , linewidths=.5 )
        #plt.show()

        print("Y_TEST values : " , y_test[:10])
        print("Y hat : " , ages_predicted[:10])

        print("Parameters : " , lr3.coef_)
        print("Intercept : " , lr3.intercept_)
        print("MSE : " , MSE)
        RMSE = numpy.sqrt(MSE)
        print("RMSE : " , RMSE)
        correlation_matrix = dataset.corr()
        print("correlation matrix:\n", correlation_matrix)
        sns.heatmap(correlation_matrix, annot=True, linewidths=.5, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

        sample_data = {
            'NPG': [1, 5, 0, 3, 4],
            'PGL': [85, 150, 85, 100, 135],
            'DIA': [66, 88, 60, 75, 82]
        }

        # Creating a DataFrame from the sample data
        sample_df = pd.DataFrame(sample_data)

        # Predicting 'AGE' using the previously trained LR3 model
        predicted_ages = lr3.predict(scaler.transform(sample_df))

        print("testttt : " , predicted_ages)

        plt.figure(figsize=(10, 6))
        plt.plot(y_test.reset_index(drop=True), label='True Values', color='green')
        plt.plot(ages_predicted, label='Predicted Values', color='red')
        plt.title('Comparison of True and Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Age')
        plt.legend()
        plt.show()

        differences = y_test - ages_predicted

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(differences)), differences, color='purple')
        plt.title('Difference Between True and Predicted Values')
        plt.xlabel('Data Point Index')
        plt.ylabel('Difference in Age')
        plt.show()
