import pandas as pd


def replaceZeros(dataset):
    columns_with_zeros = ['PGL', 'DIA', 'TSF', 'INS', 'BMI']
    for col in columns_with_zeros:
        mean=dataset[col].mean()
        dataset[col] = dataset[col].replace(0 , mean)

    return dataset

def cap_outliers(dataset , col):
    q1=dataset[col].quantile(0.25)
    q3=dataset[col].quantile(0.75)
    iqr=q3-q1
    lower_bound = q1-1.5 *iqr
    upper_bound =q3 + 1.5 * iqr
    dataset[col] = dataset[col].clip(lower=lower_bound, upper=upper_bound)



class ReadDataSet:
    def __init__(self):
        dataSET = pd.read_csv('G:\D\MachineLearning\diabetes\cleaned_diabetes_data.csv')
        #dataSET = pd.read_csv(r'C:\Users\tareq\Downloads\Diabetes.csv')

        missing_values = dataSET.isnull().sum().sum()
        if missing_values > 0:
            raise ValueError(f"Error: Original dataset contains {missing_values} missing values.")
        dataSET.info()
       # dataSET = self.cleanDataSet(dataSET)

        self.dataset = dataSET

    def cleanDataSet(self, dataset):

        datasetAfterZero = replaceZeros(dataset)
        dataSET = self.cleaner(datasetAfterZero)
       # columns_to_cap = ['NPG', 'PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE']

        #for col in columns_to_cap:
         #   cap_outliers(datasetAfterZero, col)

        dataSET.to_csv('cleaned_diabetes_data.csv', index=False)
        return dataSET
    def cleaner(self , df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (df >= lower_bound) & (df <= upper_bound)

        # Apply the mask and keep rows without outliers in any column
        df_clean = df[mask.all(axis=1)]
        return df_clean

if __name__ == '__main__':
    reader = ReadDataSet()