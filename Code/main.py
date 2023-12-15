import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from DataReader import ReadDataSet
from sklearn.preprocessing import MinMaxScaler

def categrizeAge(agesCol, diabeticTrue):
    categorizedAge = [0, 0, 0, 0, 0]
    for age, is_diabetic in zip(agesCol, diabeticTrue):
        if (age > 0 and age <= 20) and is_diabetic:
            categorizedAge[0] +=1
        elif (age > 20 and age <=40 and is_diabetic):
            categorizedAge[1] +=1
        elif (age > 40 and age <= 60 and is_diabetic):
            categorizedAge[2] +=1
        elif (age > 60 and age <= 80 and is_diabetic):
            categorizedAge[3] +=1
        elif (age > 80 and age <=100 and is_diabetic):
            categorizedAge[4] +=1
    return categorizedAge


data_reader = ReadDataSet()
dataset = data_reader.dataset
nullValues = dataset.isnull().sum()
print("Null Values:\n", nullValues)
print("______________________________________________________________________________________________________________________")
print ("descipe the Dataset : " , dataset.describe())
print("______________________________________________________________________________________________________________________")

dataset.hist( bins=30 , figsize=(50 , 50))
plt.show()


plt.figure(figsize=(20, 10))
for i, column in enumerate(dataset.columns[:-1], 1):  # Exclude 'Diabetic' column for box plots
    plt.subplot(3, 3, i)
    sns.boxplot(y=dataset[column])
    plt.title(f'Box Plot of {column}')

plt.tight_layout()
plt.show()


print("______________________________________________________________________________________________________________________")

sns.countplot(x='Diabetic' , data=dataset)
plt.title("Distribution of Diabetic")
plt.show()
print("______________________________________________________________________________________________________________________")




categrizedAge = categrizeAge(dataset['AGE'], dataset['Diabetic'])
print("categrizedAge:", categrizedAge)
print("______________________________________________________________________________________________________________________")

age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']

plt.bar(age_labels, categrizedAge)
plt.title("Distribution of Diabetic by Age Group")
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()
print("______________________________________________________________________________________________________________________")



#age_bins = [0, 20, 40, 60, 80, 100]
#age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
#dataset['AgeGroup'] = pd.cut(dataset['AGE'], bins=age_bins, labels=age_labels)

#print(dataset['AgeGroup'])
#diabetic_df = dataset[dataset['Diabetic'] == 1]


#sns.histplot(diabetic_df['AgeGroup'], bins = len(age_bins)-1 )
#plt.title('Number of Diabetics in Each Age Group')
#plt.xlabel('Age Group')
#plt.ylabel('Number of Diabetics')
#plt.show()



sns.kdeplot(x= 'AGE' , data=dataset , color='green')
plt.show()
print("______________________________________________________________________________________________________________________")

sns.kdeplot(x= 'BMI' , data=dataset , color='red')
plt.show()
print("______________________________________________________________________________________________________________________")

correlation_matrix= dataset.corr()
print("correlation matrix:\n" , correlation_matrix)
sns.heatmap(correlation_matrix , annot=True , linewidths= .5, cmap= 'coolwarm')
plt.title('Correlation Heatmap')
plt.show()
print("______________________________________________________________________________________________________________________")


sns.scatterplot(x='PGL' , y='Diabetic' , data=dataset)
plt.show()
print("______________________________________________________________________________________________________________________")


x = dataset.iloc[:,:-1]
print("X_dataset" , x.head())
print("______________________________________________________________________________________________________________________")

y=dataset.iloc[: , -1]
print("Y_dataset" , y.head())
print("______________________________________________________________________________________________________________________")
scaler = MinMaxScaler()
X_scaled_data = scaler.fit_transform(x)
X_train , X_test , y_train , y_test = train_test_split(X_scaled_data , y , test_size=.2 , random_state=8  , shuffle = True)

print("X_train" , X_train.head())
print("______________________________________________________________________________________________________________________")

print("X_test" , X_test.head())
print("______________________________________________________________________________________________________________________")


print("y_train" , y_train.head())
print("______________________________________________________________________________________________________________________")

print("y_test" , y_test.head())
print("______________________________________________________________________________________________________________________")

