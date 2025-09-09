import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the data sets using the pandas library
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

# Previewing the data sets for futhur cleaning and processing
print("Dataset1 (Bat Landings):")
# head function returns the first 5 rows of the data frame
print(dataset1.head())
# provides the detail information of the columns in the data frame
print(dataset1.info())


# Previewing the data sets for futhur cleaning and processing same as dataset 1
print("\nDataset2 (Rat Arrivals):")
print(dataset2.head())
print(dataset2.info())



# Deteming the colums and their missing values for both data ets
print("Missing values in Dataset1:\n", dataset1.isnull().sum())
print("\nMissing values in Dataset2:\n", dataset2.isnull().sum())


# Descriptive Analysis: Dataset1 - Count risk-taking vs avoidance
# Renaming the boolen values in datasets i.e. 0 is represented as 'Risk Avoidance' and 1 is represneted as 'Risk Taking'
risk_counts = dataset1['risk'].value_counts().rename({0: 'Risk Avoidance', 1: 'Risk Taking'})
# Printing the values in the terminal
print("Risk Behaviour Counts:\n", risk_counts)

plt.figure(figsize=(18,8))
# Plotting a graph by looking into risk column for occurance of different values i.e. 0 and 1 
sns.countplot(x='risk', data=dataset1, palette="Set2")
# Renaming the values for descriptive representation in the graph
plt.xticks([0,1], ['Risk Avoidance', 'Risk Taking'])
# Providing the heading name to the graph
plt.title("Bat Risk-Taking vs Avoidance")
plt.show()

reward_counts = dataset1['reward'].value_counts().rename({0: 'No Reward', 1: 'Reward'})
print("\nReward Behaviour Counts:\n", reward_counts)

# Plotting the reward behaviour and repeating the same process as for risk behaviour
plt.figure(figsize=(18,8))
sns.countplot(x='reward', data=dataset1, palette="Set1")
plt.xticks([0,1], ['No Reward', 'Reward'])
plt.title("Bat Reward Behaviour")
plt.show()

# Plotting the risk behaviour by season and repeating the same process as for risk behaviour
# Risk behaviour by season
plt.figure(figsize=(18,8))
sns.countplot(x='season', hue='risk', data=dataset1, palette="muted")
plt.xticks(rotation=30)
plt.legend(title="Risk (0=Avoid,1=Take)")
plt.title("Risk Behaviour Across Seasons")
plt.show()

plt.figure(figsize=(18,8))
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=dataset2)
plt.title("Relationship Between Rat Arrivals and Bat Landings")
plt.xlabel("Number of Rat Arrivals")
plt.ylabel("Number of Bat Landings")
plt.show()

# Line plot: Food availability over time
plt.figure(figsize=(18,8))
sns.lineplot(x='time', y='food_availability', data=dataset2)
plt.xticks(rotation=45)
plt.title("Food Availability Over Time")
plt.show()