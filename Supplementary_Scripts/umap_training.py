import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

#read csv file
flow1 = pd.read_csv('oneHot_clustering.csv')

features = ['Waveform_type_biphasic_asymmetric', 'Waveform_type_biphasic_balanced',
'Waveform_type_biphasic_capacitive', 'Waveform_type_monophasic', 'GSA', 'Pulse_width',
'Frequency', 'Current', 'Voltage', 'Charge_per_phase', 'Charge_density', 'Current_density',
'stim_on', 'stim_total_day', 'daily_pulses', 'daily_accumulated_charge']

input = flow1.loc[:, features].values
output = flow1.loc[:, ['Avg_Damage_binary']].values
input = StandardScaler().fit_transform(input)

umap_2d = UMAP(n_components=2, init='random', random_state=0)

input_embedded = umap_2d.fit_transform(input)
label1 = input_embedded[:, 0]
label2 = input_embedded[:, 1]

flow1['LABEL1'] = label1
flow1['LABEL2'] = label2

flow1.to_csv(r'umap_results.csv', index=None, header=True)