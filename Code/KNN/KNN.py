from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


from DataReader import ReadDataSet
class KNN_Task :
    @staticmethod
    def predict() :
        datareader = ReadDataSet()
        dataset = datareader.dataset
        x = dataset.iloc[:,:-1]

        x = dataset[['PGL' , 'BMI' , 'AGE' , 'NPG']]
        print(" X features : ",  x.head())
        y= dataset.iloc[:,-1]
        print("Y feature : " , y.head())
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        x_train , x_test , y_train, y_test = train_test_split(x , y , test_size=0.2 , random_state=0)
        knn = KNeighborsClassifier()
        knn.fit(x_train , y_train)
        y_pred = knn.predict(x_test)

        sns.scatterplot(x = dataset['PGL'] , y=dataset['BMI'] , hue=dataset['Diabetic'])
        plt.title('Scatter Plot of PGL vs BMI by Diabetic Status')
        plt.xlabel('Plasma Glucose Level (PGL)')
        plt.ylabel('Body Mass Index (BMI)')
        plt.legend(title='Diabetic Status')
        plt.show()




        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy * 100}%")



    @staticmethod
    def ModelK4():
        datareader = ReadDataSet()
        dataset = datareader.dataset
        #x = dataset.iloc[:, :-1]
        x = dataset[['PGL' , 'BMI' , 'AGE' , 'NPG']]

        print(" X features : ", x.head())
        y = dataset.iloc[:, -1]
        print("Y feature : ", y.head())
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100}%")

    @staticmethod
    def ModelK7():
        datareader = ReadDataSet()
        dataset = datareader.dataset
        #x = dataset.iloc[:, :-1]
        x = dataset[['PGL' , 'BMI' , 'AGE' , 'NPG']]

        print(" X features : ", x.head())
        y = dataset.iloc[:, -1]
        print("Y feature : ", y.head())
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100}%")

    @staticmethod
    def ModelK10():
        datareader = ReadDataSet()
        dataset = datareader.dataset
        #x = dataset.iloc[:, :-1]
        x = dataset[['PGL' , 'BMI' , 'AGE' , 'NPG' ]]
        print(" X features : ", x.head())
        y = dataset.iloc[:, -1]
        print("Y feature : ", y.head())
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100}%")

    @staticmethod
    def ModelK15():
        datareader = ReadDataSet()
        dataset = datareader.dataset
        #x = dataset.iloc[:, :-1]
        x = dataset[['PGL' , 'BMI' , 'AGE' , 'NPG']]

        print(" X features : ", x.head())
        y = dataset.iloc[:, -1]
        print("Y feature : ", y.head())
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100}%")

    @staticmethod
    def ModelK25():
        datareader = ReadDataSet()
        dataset = datareader.dataset
        #x = dataset.iloc[:, :-1]
        x = dataset[['PGL' , 'BMI' , 'AGE' , 'NPG']]

        print(" X features : ", x.head())
        y = dataset.iloc[:, -1]
        print("Y feature : ", y.head())
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        knn = KNeighborsClassifier(n_neighbors=25)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100}%")




