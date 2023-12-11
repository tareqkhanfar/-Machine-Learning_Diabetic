import pandas as pd

class ReadDataSet:
    def __init__(self):
        dataSET = pd.read_csv('G:\D\MachineLearning\diabetes\dataset\Diabetes.csv')
        missing_values = dataSET.isnull().sum().sum()
        if missing_values > 0:
            raise ValueError(f"Error: Original dataset contains {missing_values} missing values.")
        dataSET.info()

        self.dataset = dataSET

