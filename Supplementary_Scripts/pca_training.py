import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#read csv file
flow1 = pd.read_csv('oneHot_clustering.csv')

features = ['Waveform_type_biphasic_asymmetric', 'Waveform_type_biphasic_balanced',
'Waveform_type_biphasic_capacitive', 'Waveform_type_monophasic', 'GSA', 'Pulse_width',
'Frequency', 'Current', 'Voltage', 'Charge_per_phase', 'Charge_density', 'Current_density',
'stim_on', 'stim_total_day', 'daily_pulses', 'daily_accumulated_charge']

input = flow1.loc[:, features].values
output = flow1.loc[:, ['Avg_Damage_binary']].values
input = StandardScaler().fit_transform(input)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(input)
print(pca.explained_variance_ratio_.cumsum())

label1 = principalComponents[:, 0]
label2 = principalComponents[:, 1]

flow1['LABEL1'] = label1
flow1['LABEL2'] = label2

flow1.to_csv(r'pca_results.csv', index=None, header=True)