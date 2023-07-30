import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt

#read csv file
flow1 = pd.read_csv('original_data.csv')

binary = flow1['Avg_Damage_binary'].values
shannon = flow1['k'].values

max_accuracy = 0
thresh = 0

acc = []
thr = []

for i in range(40000):
    threshold = i * 0.0001
    thr.append(threshold)

    prediction = []
    for j in range(len(shannon)):
        if (shannon[j] > threshold):
            prediction.append(1)
        else:
            prediction.append(0)

    # Model Accuracy, how often is the classifier correct?
    accuracy = accuracy_score(binary, prediction)
    acc.append(accuracy)

    if (accuracy>max_accuracy):
        thresh = threshold
        max_accuracy = accuracy

print(str(thresh))
print(str(max_accuracy))

thr1 = np.asarray(thr)
acc1 = np.asarray(acc)

plt.scatter(thr1, acc1)
plt.show()

output = pd.DataFrame({'k threshold': thr, 'accuracy': acc})
output.to_csv('Shannon_equation.csv', index=False)






