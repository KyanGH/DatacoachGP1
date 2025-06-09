import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

# Load data and print information about the data
df = pd.read_csv('SVMtrain.csv')
data_np = df.to_numpy()
print(f'Data shape: {data_np.shape}')

# Scatter Plot: Age vs Survived
ages = data_np[:,4]
survived = data_np[:,1]
plt.scatter(ages, survived, color='red')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Age vs Survival')
plt.grid(True)
plt.show()

# Histogram Plot: ages survived
ages_survived = data_np[data_np[:, 1] == 1, 4]
plt.hist(ages_survived, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of survivors')
plt.title('Age Distribution of Survivors')
plt.show()