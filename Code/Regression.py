import numpy
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

        x_train , x_test , y_train , y_test = train_test_split(X_scaled_data , y , test_size=0.2 , random_state=8 , shuffle=True )
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

    @staticmethod
    def LR2() :
        data = ReadDataSet()
        dataset = data.dataset
        x = dataset['NPG'].values.reshape(-1, 1)  # Reshape to 2D array
        print(x[:5])
        y = dataset['AGE']


        x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=8 , shuffle=True )
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


