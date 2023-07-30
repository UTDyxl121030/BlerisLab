import matplotlib.pyplot as plt
import pandas as pd

#read csv file
flow1 = pd.read_csv('original_data.csv')


#get the Series for each of the columns
Charge_per_phase = flow1["Charge_per_phase"]
Charge_density_per_phase = flow1["Charge_density"]

fig1, ax1 = plt.subplots()
ax1.set_title('Feature Distributions')
data = [Charge_per_phase, Charge_density]

ax1.boxplot(data, showfliers=False)
ax1.set_xticklabels(['Charge_per_phase', 'Charge_density_per_phase'])

plt.show()

print(flow1.mean())
print(flow1.std())


