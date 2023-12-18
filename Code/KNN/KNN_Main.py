from matplotlib import pyplot as plt

from KNN import KNN_Task
import seaborn as sns

#KNN_Task.predict()


metrics_summary, conf_matrix = KNN_Task.ModelK10()

# Print the results
print("Metrics Summary:")
print((metrics_summary))
print("\nConfusion Matrix:")
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()