import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, pearsonr, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (8,5)

dataset1 = pd.read_csv("data/dataset1.csv")
dataset2 = pd.read_csv("data/dataset2.csv")

print("Dataset1 shape:", dataset1.shape)
print("Dataset2 shape:", dataset2.shape)

dataset1.columns = [c.strip().lower() for c in dataset1.columns]
dataset2.columns = [c.strip().lower() for c in dataset2.columns]

print("\nDataset1 columns:", dataset1.columns.tolist())
print("Dataset2 columns:", dataset2.columns.tolist())

risk_col = 'risk'
reward_col = 'reward'
season_col = 'season'
rat_arrivals_col = 'rat_arrival_number'
bat_landings_col = 'bat_landing_number'
food_col = 'food_availability'
time_col_d1 = 'start_time'   
time_col_d2 = 'time'

for col in [risk_col, reward_col]:
    if col in dataset1.columns:
        dataset1[col] = pd.to_numeric(dataset1[col], errors='coerce').astype('Int64')

for col in [rat_arrivals_col, bat_landings_col, food_col]:
    if col in dataset2.columns:
        dataset2[col] = pd.to_numeric(dataset2[col], errors='coerce')

if season_col in dataset1.columns:
    dataset1[season_col] = dataset1[season_col].astype(str).str.strip().str.lower()

for c, df in [(time_col_d1, dataset1), (time_col_d2, dataset2)]:
    if c in df.columns:
        try:
            df[c] = pd.to_datetime(df[c])
        except:
            pass


if risk_col in dataset1.columns:
    before = dataset1.shape
    dataset1 = dataset1.dropna(subset=[risk_col])
    print(f"\nDropped rows with missing '{risk_col}' in dataset1: {before} -> {dataset1.shape}")

if (rat_arrivals_col in dataset2.columns) and (bat_landings_col in dataset2.columns):
    before = dataset2.shape
    dataset2 = dataset2.dropna(subset=[rat_arrivals_col, bat_landings_col])
    print(f"Dropped rows with missing '{rat_arrivals_col}'/'{bat_landings_col}' in dataset2: {before} -> {dataset2.shape}")

print("\nAfter cleaning - shapes:")
print(" - dataset1:", dataset1.shape)
print(" - dataset2:", dataset2.shape)

if rat_arrivals_col in dataset2.columns:
    dataset2['rat_present'] = (dataset2[rat_arrivals_col] > 0).astype(int)
    def rat_intensity(x):
        if pd.isna(x): return np.nan
        if x == 0: return 0
        if x <= 2: return 1
        if x <= 5: return 2
        return 3
    dataset2['rat_intensity'] = dataset2[rat_arrivals_col].apply(rat_intensity)

if season_col not in dataset1.columns and 'month' in dataset1.columns:
    dataset1['season'] = dataset1['month'].map({
        12:'summer',1:'summer',2:'summer',3:'autumn',4:'autumn',5:'autumn',
        6:'winter',7:'winter',8:'winter',9:'spring',10:'spring',11:'spring'
    })

if season_col not in dataset2.columns and 'month' in dataset2.columns:
    dataset2['season'] = dataset2['month'].map({
        12:'summer',1:'summer',2:'summer',3:'autumn',4:'autumn',5:'autumn',
        6:'winter',7:'winter',8:'winter',9:'spring',10:'spring',11:'spring'
    })

def floor_to_30min(dt):
    return dt.dt.floor('30T')

if (time_col_d1 in dataset1.columns) and (time_col_d2 in dataset2.columns):
    dataset1['interval'] = floor_to_30min(dataset1[time_col_d1])
    dataset2['interval'] = floor_to_30min(dataset2[time_col_d2])
    merged = dataset1.merge(dataset2[['interval','rat_present',rat_arrivals_col,
                                      'rat_intensity',food_col,bat_landings_col]],
                            on='interval', how='left')
else:
    merged = dataset1.copy()
    merged['rat_present'] = np.nan

print("Merged dataset shape:", merged.shape)

# Investigation A: Risk-taking vs rat presence
if 'risk' in merged.columns and 'rat_present' in merged.columns:
    temp = merged.dropna(subset=['risk','rat_present'])
    if len(temp) > 0:
        rat_risk_ct = pd.crosstab(temp['rat_present'], temp['risk'])
        print("Contingency Table (Rat Presence x Risk):\n", rat_risk_ct)

        prop_by_rat = temp.groupby('rat_present')['risk'].mean().reset_index()
        if not prop_by_rat.empty:
            sns.barplot(x='rat_present', y='risk', data=prop_by_rat)
            plt.xticks([0,1], ['No Rats', 'Rats Present'])
            plt.ylabel('Proportion Risk-taking')
            plt.title('Risk-taking vs Rat Presence')
            plt.show()

# Investigation A: Rat arrivals vs Bat landings
if rat_arrivals_col in dataset2.columns and bat_landings_col in dataset2.columns:
    temp = dataset2.dropna(subset=[rat_arrivals_col, bat_landings_col])
    if len(temp) > 0:
        sns.scatterplot(x=rat_arrivals_col, y=bat_landings_col, data=temp, alpha=0.6)
        plt.title('Rat Arrivals vs Bat Landings')
        plt.show()

# Investigation B: Risk-taking by season
if 'season' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['season','risk'])
    if len(temp) > 0:
        sns.barplot(x='season', y='risk', data=temp, estimator=np.mean,
                    order=sorted(temp['season'].unique()))
        plt.title('Risk-taking by Season')
        plt.ylabel('Proportion Risk-taking')
        plt.show()

# Investigation B: Bat landings by season
if 'season' in dataset2.columns and 'bat_landing_number' in dataset2.columns:
    temp = dataset2.dropna(subset=['season','bat_landing_number'])
    if len(temp) > 0:
        sns.boxplot(x='season', y='bat_landing_number', data=temp,
                    order=sorted(temp['season'].unique()))
        plt.title('Bat Landings by Season')
        plt.show()

# Investigation A: Chi-square test
if 'rat_present' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['rat_present','risk'])
    if len(temp) > 0:
        contingency = pd.crosstab(temp['rat_present'], temp['risk'])
        if contingency.shape[0] > 0 and contingency.shape[1] > 0:
            chi2, p, dof, expected = chi2_contingency(contingency.fillna(0))
            print(f"Chi-square: chi2={chi2:.3f}, dof={dof}, p={p:.4f}")

# Investigation A: Proportions z-test
if 'rat_present' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['rat_present','risk'])
    grouped = temp.groupby('rat_present')['risk'].agg(['sum','count']).reset_index()
    if grouped.shape[0] == 2:
        stat, pval = proportions_ztest(grouped['sum'].values, grouped['count'].values)
        print(f"Proportions z-test: stat={stat:.3f}, p={pval:.4f}")

# Investigation A: Correlation
if rat_arrivals_col in dataset2.columns and bat_landings_col in dataset2.columns:
    temp = dataset2.dropna(subset=[rat_arrivals_col, bat_landings_col])
    if len(temp) > 1:
        r, p = pearsonr(temp[rat_arrivals_col], temp[bat_landings_col])
        print(f"Pearson correlation: r={r:.3f}, p={p:.4f}")

# Investigation B: Seasonal risk-taking (Winter vs Spring)
if 'season' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['season','risk'])
    if {'winter','spring'}.issubset(set(temp['season'].unique())):
        winter = temp[temp['season']=='winter']['risk']
        spring = temp[temp['season']=='spring']['risk']
        if len(winter) > 0 and len(spring) > 0:
            stat, pval = proportions_ztest([winter.sum(), spring.sum()],
                                           [len(winter), len(spring)])
            print(f"Winter vs Spring risk-taking z-test: stat={stat:.3f}, p={pval:.4f}")

# Investigation B: Seasonal bat landings
if 'season' in dataset2.columns and 'bat_landing_number' in dataset2.columns:
    temp = dataset2.dropna(subset=['season','bat_landing_number'])
    if {'winter','spring'}.issubset(set(temp['season'].unique())):
        w = temp[temp['season']=='winter']['bat_landing_number']
        s = temp[temp['season']=='spring']['bat_landing_number']
        if len(w) > 1 and len(s) > 1:
            tstat, tp = ttest_ind(w, s, equal_var=False, nan_policy='omit')
            print(f"Winter vs Spring bat landings t-test: t={tstat:.3f}, p={tp:.4f}")

print("\n--- Logistic Regression (risk ~ rat_present + food + season) ---")

logit = None
if ('risk' in merged.columns) and ('rat_present' in merged.columns):
    model_df = merged[['risk', 'rat_present']].copy()
    if food_col in merged.columns:
        model_df['food_avail'] = merged[food_col]
    if 'season' in merged.columns:
        model_df['season_cat'] = merged['season'].astype('category')

    model_df = model_df.dropna()
    print("Model dataframe shape:", model_df.shape)
    print("Unique 'risk' values:", model_df['risk'].unique())

    if model_df['risk'].nunique() < 2:
        print("Model cannot be fit: 'risk' only has one unique value.")
    else:
        formula = 'risk ~ rat_present'
        if 'food_avail' in model_df.columns:
            formula += ' + food_avail'
        if 'season_cat' in model_df.columns:
            formula += ' + C(season_cat)'

        print("Using formula:", formula)

        try:
            logit = smf.logit(formula=formula, data=model_df).fit(disp=False)
            print("\n✅ Logistic regression fitted successfully!\n")
            print(logit.summary())
        except Exception as e:
            print("Logistic regression error:", e)
else:
    print("Skipping logistic regression — required columns missing.")


if rat_arrivals_col in dataset2.columns and bat_landings_col in dataset2.columns:
    dfm = dataset2.dropna(subset=[rat_arrivals_col, bat_landings_col, food_col]).copy()
    if len(dfm) > 0:
        formula = f"{bat_landings_col} ~ {rat_arrivals_col} + {food_col} + C(season)"
        try:
            poisson_model = smf.glm(formula=formula, data=dfm,
                                    family=sm.families.Poisson()).fit()
            print(poisson_model.summary())
        except Exception as e:
            print("Poisson model error:", e)


if 'season' in merged.columns and 'risk' in merged.columns:
    temp = merged.dropna(subset=['season','risk'])
    if len(temp) > 0:
        sns.barplot(x='season', y='risk', data=temp, estimator=np.mean,
                    order=sorted(temp['season'].unique()))
        plt.title('Risk-taking across Seasons')
        plt.show()

if 'season' in dataset2.columns and 'bat_landing_number' in dataset2.columns:
    temp = dataset2.dropna(subset=['season','bat_landing_number'])
    if len(temp) > 0:
        sns.boxplot(x='season', y='bat_landing_number', data=temp,
                    order=sorted(temp['season'].unique()))
        plt.title('Bat Landings across Seasons')
        plt.show()

if 'season' in dataset2.columns and 'food_availability' in dataset2.columns:
    temp = dataset2.dropna(subset=['season','food_availability'])
    if len(temp) > 0:
        sns.boxplot(x='season', y='food_availability', data=temp,
                    order=sorted(temp['season'].unique()))
        plt.title('Food Availability across Seasons')
        plt.show()

try:
    with open("logistic_summary.txt","w") as f:
        f.write(logit.summary().as_text())
    print("Saved logistic_summary.txt")
except:
    print("Logistic regression summary not available")

try:
    with open("poisson_summary.txt","w") as f:
        f.write(poisson_model.summary().as_text())
    print("Saved poisson_summary.txt")
except:
    print("Poisson regression summary not available")

if 'risk' in merged.columns:
    plt.figure()
    sns.countplot(x='risk', data=merged)
    plt.title("Risk-taking vs Avoidance (Overall)")
    plt.savefig("risk_counts.png")

if rat_arrivals_col in dataset2.columns and bat_landings_col in dataset2.columns:
    plt.figure()
    sns.scatterplot(x=rat_arrivals_col, y=bat_landings_col, data=dataset2, alpha=0.6)
    plt.title("Rat Arrivals vs Bat Landings")
    plt.savefig("rat_vs_bats.png")

if 'season' in merged.columns and 'risk' in merged.columns:
    plt.figure()
    sns.barplot(x='season', y='risk', data=merged, estimator=np.mean,
                order=sorted(merged['season'].dropna().unique()))
    plt.title("Risk-taking across Seasons")
    plt.savefig("risk_by_season.png")

print("Figures and model summaries saved (where available).")





