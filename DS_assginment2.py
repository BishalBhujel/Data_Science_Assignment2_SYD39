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