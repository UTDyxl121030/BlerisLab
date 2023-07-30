import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

#read csv file
flow1 = pd.read_csv('new_oneHot_training.csv')

#build the DataFrame for the features: X
input = flow1[['Waveform_type_biphasic_asymmetric', 'Waveform_type_biphasic_balanced',
'Waveform_type_biphasic_capacitive', 'Waveform_type_monophasic', 'GSA', 'Pulse_width',
'Frequency', 'Current', 'Voltage', 'Charge_per_phase', 'Charge_density', 'Current_density',
'stim_on', 'stim_total_day', 'daily_pulses', 'daily_accumulated_charge']]


#build the DataFrame for the label Avg_Damage_binary: Y
y = flow1[['Avg_Damage_binary']]
le = LabelEncoder()
le.fit(y)
output = le.transform(y)

logReg = LogisticRegression()


###Perform 10-fold CV
scores = cross_val_score(logReg, input, output, cv=10)
#print(type(scores4))
mean1 = np.mean(scores)

print(scores)
print(str(mean1))
print(str(np.std(scores)))


###Build the model
logReg.fit(input, output)

