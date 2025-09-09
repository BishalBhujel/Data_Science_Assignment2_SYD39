'''Investigation A: Do Bats Perceive Rats as Predators?

This notebook addresses Objective 1 (Assessment 2) of the HIT140 Foundations of Data Science project.

We investigate whether bats perceive rats only as food competitors or also as potential predators.  
If rats are considered a predation risk, bats should show more avoidance behaviour or increased vigilance during foraging.

We use two datasets:
- dataset1.csv → Bat landings & behaviours (risk, reward, season, etc.)  
- dataset2.csv → Rat arrivals & bat activity (arrivals, food availability, etc.)

We will perform both:
1. Descriptive analysis (counts, trends, plots)  
2. Inferential analysis (statistical tests: chi-square, correlation)
'''


'''Importing libraries
    1. Pandas to process the provided data in dataset1 and dataset2
    2. matplotlib for plotting and visualization
    3. seaborn is a highlevel interface built on top of matplotlib which can work directly on pandas df
    4. scipy is used to perform scientific calculation such as correlation, chi square test etc.
'''    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr

# Configuring the specific style i.e graph background is white, muted colors and bigger font size
sns.set(style="whitegrid", palette="muted", font_scale=1.1)


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

# Plotting the risk behaviour
# Defining the size of the graph
plt.figure(figsize=(18,8))
# Plotting a graph by looking into risk column for occurance of different values i.e. 0 and 1 
sns.countplot(x='risk', data=dataset1, palette="Set2")
# Renaming the values for descriptive representation in the graph
plt.xticks([0,1], ['Risk Avoidance', 'Risk Taking'])
# Providing the heading name to the graph
plt.title("Bat Risk-Taking vs Avoidance")
plt.show()

# Renaming the boolen values in datasets i.e. 0 is represented as 'No Reward' and 1 is represneted as 'Reward'
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


# Descriptive Analysis: Dataset2 - Scatter plot: Rat arrivals vs Bat landings
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


# Inferential Analysis
# 1. Chi-Square Test: Risk vs Reward

# Create contingency table
contingency_table = pd.crosstab(dataset1['risk'], dataset1['reward'])
print("Contingency Table (Risk vs Reward):\n")
print(contingency_table, "\n")

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square Test Results (Risk vs Reward):")
print("Chi2 statistic:", chi2)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)
print("p-value:", p)

if p < 0.05:
    print("\n✅ Significant relationship between Risk and Reward (p < 0.05)")
else:
    print("\n❌ No significant relationship between Risk and Reward (p >= 0.05)")


# 2. Correlation Test: Rat Arrivals vs Bat Landings
print("\nCorrelation Test (Rat arrivals vs Bat landings):")

corr, pval = pearsonr(dataset2['rat_arrival_number'], dataset2['bat_landing_number'])

print("Correlation coefficient (r):", corr)
print("p-value:", pval)

if pval < 0.05:
    print("✅ Significant correlation between Rat arrivals and Bat landings (p < 0.05)")
else:
    print("❌ No significant correlation (p >= 0.05)")


# # Conclusion
# 
# From the descriptive and inferential analysis:
# 
# Risk Behaviour: The majority of bats showed risk avoidance, but a portion engaged in risk taking.  
# Reward Behaviour: Reward outcomes varied, suggesting risk-taking does not always guarantee food access.  
# Seasonal Patterns: Risk-taking appeared to differ across seasons, possibly influenced by food availability.  
# Rat vs Bat Relationship: Scatterplots and correlation tests indicated whether increased rat arrivals reduce bat landings.  
# 
# Interpretation:  
# If bats consistently avoid feeding when rats are present, it supports the hypothesis that bats perceive rats as predators (not just competitors).  
# If there is no significant difference, then rats may be seen mainly as competitors.  
# 
# This evidence contributes to Investigation A and addresses Objective 1 (Assessment 2).
# 
