import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

#read csv file
data1 = pd.read_csv('original_data.csv')

#build the DataFrame for the features: X
X_categorical = data1[['Animal_model', 'Nervous_system', 'Specific_target', 'Geometry', 'Location', 'Material',
           'charge_mechanism', 'Waveform_type', 'first_ph_direction', 'Polarity', 'Control', 'Bias']]

X_numerical = data1[['GSA', 'Pulse_width', 'Frequency', 'Interpulse_delay', 'Current', 'Voltage', 'Charge_per_phase',
                     'Charge_density', 'Current_density', 'Duty_cycle', 'stim_on', 'stim_off', 'stim_total_day', 'days-stim',
                     'stim_total_study', 'daily_pulses', 'total_pulses', 'daily_accumulated_charge', 'total_accumulated_charge']]

#pre-process categorical data using ordinal encoding

oe = OrdinalEncoder()
oe.fit(X_categorical)
X_categorical_enc = oe.transform(X_categorical)
X_categorical_enc = pd.DataFrame(X_categorical_enc, columns=['Animal_model', 'Nervous_system', 'Specific_target', 'Geometry', 'Location', 'Material',
           'charge_mechanism', 'Waveform_type', 'first_ph_direction', 'Polarity', 'Control', 'Bias'])

#build the X dataframe
X_enc = pd.concat([X_categorical_enc, X_numerical], axis=1)

features = list(X_enc.columns)

#build the DataFrame for the label Avg_Damage_binary: Y
Y = data1[['Avg_Damage_binary']]

le = LabelEncoder()
le.fit(Y)
Y_enc = le.transform(Y)


###Perform Importance Ranking using Avg_Damage_binary

Bforest = ExtraTreesClassifier(n_estimators=250, random_state=0)
Bforest.fit(X_enc, Y_enc)
Bimportance = Bforest.feature_importances_

for i, v in enumerate(Bimportance):
    #print(str(i))
    print('Feature: {}  Score: {}' .format(i, v))

# plot feature importance
pyplot.bar([x for x in range(len(Bimportance))], Bimportance)
pyplot.show()

data = {'Features': features, 'ordinal_binary_randomForest': Bimportance}

df = pd.DataFrame(data)

df.to_csv('importance_results.csv', index=False)