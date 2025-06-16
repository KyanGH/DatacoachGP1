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

# Plot for survivors based on Gender
genders_survived = data_np[data_np[:, 1] == 1, 3]
genders = np.array(['Male','Female'])
male_count = np.sum(genders_survived == 'Male')
female_count = np.sum(genders_survived == 'female')
survivors_count = [male_count, female_count]
plt.bar(genders, survivors_count, color = ['red', 'blue'], edgecolor='black', width=0.2)
plt.xlabel('Gender')
plt.ylabel('Number of survivors')
plt.title('Gender vs Survival')
plt.show()
print(genders_survived.shape)
print(genders_survived[:10])