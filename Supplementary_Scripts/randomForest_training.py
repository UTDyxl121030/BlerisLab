import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
import joblib
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

index = []
mean = []
std = []

for i in range(100):
    temp1 = i + 1
    print(str(temp1))
    name1 = 'C:\\Path_to_folder\\rf_cv10_' + str(temp1) + '.pkl'
    
    #Create a random forest Classifier
    rf = RandomForestClassifier(n_estimators=temp1)

    scores = cross_val_score(rf, input, output, cv=10)
    mean1 = np.mean(scores)

    if (mean1 > 0.0):
        print(scores)
        print(str(mean1))
        print(str(np.std(scores)))

        index.append(temp1)
        mean.append(str(mean1))
        std.append(str(np.std(scores)))

        # Save the model as a pickle in a file
        rf.fit(input, output)
        joblib.dump(rf, name1)

output = pd.DataFrame({'tree number': index, 'mean': mean, 'std': std})
output.to_csv('rf_cv.csv', index=False)

