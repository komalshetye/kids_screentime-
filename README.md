#kids screen time 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/Komal/Desktop/data analyst/Indian_Kids_Screen_Time.
df.info()
<class 'pandas.core.frame.DataFrame'> RangeIndex: 9712 entries, 0 to 9711 Data columns (total 8 columns):
#	Column	Non-Null Count  Dtype

 	 
0	Age	9712	non-null		 
int64
1	Gender	9712	non-null		object
2	Avg_Daily_Screen_Time_hr	9712	non-null		float64
3	Primary_Device	9712	non-null		object
4	Exceeded_Recommended_Limit	9712	non-null		bool
5	Educational_to_Recreational_Ratio	9712	non-null		float64
6	Health_Impacts	6494	non-null		object
7	Urban_or_Rural	9712	non-null		object
dtypes: bool(1), float64(2), int64(1), object(4) memory usage: 540.7+ KB
df.shape
(9712, 8)

df.head()
Age  Gender  Avg_Daily_Screen_Time_hr  Primary_Device  Exceeded_Recomme

0	14	Male	3.99	Smartphone
 

 
2	18	Female	3.73	TV

4	12	Female	5.89	Smartphone

import missingno as msno
from warnings import filterwarnings

filterwarnings(action='ignore')
msno.matrix(df)
<img width="983" height="518" alt="image" src="https://github.com/user-attachments/assets/2ba3c42e-b64e-414d-8289-b9c545ffd157" />

msno.matrix( df,
figsize=(12, 6),	# Change figure size color=(0.3, 0.5, 0.8), # Set a custom color (R, G, B) fontsize=12,	# Adjust font size of labels
sparkline=True,	# Show the sparkline at the right
labels=True	# Show column labels
)
<img width="951" height="604" alt="image" src="https://github.com/user-attachments/assets/d206c51d-8770-4ef9-9975-59b05e90c0e3" />

df.isnull().sum()
Age	0
Gender	0
Avg_Daily_Screen_Time_hr	0
Primary_Device	0
Exceeded_Recommended_Limit	0
Educational_to_Recreational_Ratio	0
Health_Impacts	3218
Urban_or_Rural	0
dtype: int64	

df.isnull().any()
Age	False
Gender	False
Avg_Daily_Screen_Time_hr	False
Primary_Device	False
Exceeded_Recommended_Limit	False
Educational_to_Recreational_Ratio	False
Health_Impacts	True
Urban_or_Rural	False
dtype: bool	

msno.bar(df)
<img width="959" height="528" alt="image" src="https://github.com/user-attachments/assets/872cee48-7281-4741-ab58-15ec0484cd26" />

df['Health_Impacts'].value_counts()
Out[22]:	Health_Impacts	
	Poor Sleep	2268
	Poor Sleep, Eye Strain	979
	Eye Strain	644
	Poor Sleep, Anxiety	608
	Poor Sleep, Obesity Risk	452
	Anxiety	385
 Poor Sleep, Eye Strain, Anxiety	258
Obesity Risk	252
Poor Sleep, Eye Strain, Obesity Risk	188
Eye Strain, Anxiety	135
Eye Strain, Obesity Risk	106
Poor Sleep, Anxiety, Obesity Risk	78
Anxiety, Obesity Risk		69
Poor Sleep, Eye Strain, Anxiety, Obesity	Risk	37
Eye Strain, Anxiety, Obesity Risk		35
Name: count, dtype: int64		

df['Health_Impacts'].fillna('Missing',inplace=True)

df['Health_Impacts'].value_counts().plot(kind='bar')
<img width="847" height="1113" alt="image" src="https://github.com/user-attachments/assets/ef7a66bc-4b76-4450-836f-2dadfdb3b0b1" />

df['Health_Impacts'].value_counts().plot(kind='kde')

<img width="909" height="611" alt="image" src="https://github.com/user-attachments/assets/08b021d8-3e43-4697-ad32-1f11fc407417" />

df['Health_Impacts'].value_counts().plot(kind='pie')
<img width="957" height="407" alt="image" src="https://github.com/user-attachments/assets/9aff7f8a-7380-47ce-bf6b-05c0700811d7" />

df['Age'].value_counts()
Age	
17	919
8	912
13	910
14	896
9	885
10	877
16	876
12	867
11	866
15	864
18	840
Name:	count, dtype: int64

f['Age'].isnull().sum()
0
df['Age'].value_counts().plot(kind='pie')
<img width="589" height="519" alt="image" src="https://github.com/user-attachments/assets/cf9956e2-3b8f-477b-8fab-0bc4c54df444" />

df['Age'].value_counts().plot(kind='barh')
<img width="852" height="611" alt="image" src="https://github.com/user-attachments/assets/8860b326-3e67-4afd-a1fa-29f00b3b3d33" />

goupby=df.groupby(['Age','Health_Impacts']).size().reset_index(name='counts')
print(goupby.head())
Age	Health_Impacts	counts
8	Anxiety	26
8	Anxiety, Obesity Risk	3
8	Eye Strain	42
8	Eye Strain, Anxiety	7
8	Eye Strain, Obesity Risk	8

import seaborn as sns
import matplotlib.pyplot as plt

# First, check if there's a typo in the variable name # It should be 'groupby' instead of 'goupby'

# Option 1: If you need to create a count column first
# Assuming you want to count occurrences by Age and Health_Impacts
groupby = df.groupby(['Age', 'Health_Impacts']).size().reset_index(name='Count

# Option 2: If you want to use an existing column instead of 'Count'
# Replace 'YourExistingColumn' with the actual column name in your dataframe
# sns.barplot(x='Age', y='YourExistingColumn', hue='Health_Impacts', data=grou

sns.barplot(x='Age', y='Count', hue='Health_Impacts', data=groupby) plt.title('Health Impacts by Age')
plt.xlabel('Age')
plt.ylabel('Count') plt.show()
<img width="864" height="677" alt="image" src="https://github.com/user-attachments/assets/df23c826-1808-4baa-9c4c-f245e16bfdb5" />


pivot = df.pivot_table(index='Age', columns='Health_Impacts', aggfunc='size', sns.heatmap(pivot, cmap='YlGnBu', annot=True)
plt.title('Health Impacts by Age (Heatmap)') plt.show()
<img width="811" height="1141" alt="image" src="https://github.com/user-attachments/assets/9adbba5a-8dcc-4cdd-89e0-fdfcef3e06c2" />

df['Avg_Daily_Screen_Time_hr'].describe()
count	9712.000000
mean	4.352837
std	1.718232
min	0.000000
25%	3.410000
50%	4.440000
75%	5.380000
max	13.890000
Name:	Avg_Daily_Screen_Time_hr, dtype: float64


sns.catplot(
data=df, x='Avg_Daily_Screen_Time_hr', kind='boxen',
)
<img width="752" height="739" alt="image" src="https://github.com/user-attachments/assets/98831d48-1636-4d7f-9e66-a7cc568efa35" />

df['Age'].describe()
count	9712.000000
mean	12.979201
std	3.162437
min	8.000000
25%	10.000000
50%	13.000000
75%	16.000000
max	18.000000
Name:	Age, dtype: float64

sns.histplot(
df, x=df['Avg_Daily_Screen_Time_hr'], hue='Gender',
kde=True,

)
<img width="864" height="649" alt="image" src="https://github.com/user-attachments/assets/66c167a9-6fb7-4bcb-96f3-3b5067a9bc61" />

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as mc

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams["figure.figsize"]=(8,5) 
plt.rcParams["axes.grid"]=True 
plt.show()

df.info()
<class 'pandas.core.frame.DataFrame'> RangeIndex: 9712 entries, 0 to 9711 Data columns (total 8 columns):
#	Column	Non-Null Count  Dtype

 	 
0	Age	9712	non-null		 
int64
1	Gender	9712	non-null		object
2	Avg_Daily_Screen_Time_hr	9712	non-null		float64
3	Primary_Device	9712	non-null		object
4	Exceeded_Recommended_Limit	9712	non-null		bool
5	Educational_to_Recreational_Ratio	9712	non-null		float64
6	Health_Impacts	6494	non-null		object
7	Urban_or_Rural	9712	non-null		object
dtypes: bool(1), float64(2), int64(1), object(4) memory usage: 540.7+ KB
## Here, cat_cols lists all columns that represent categories. #The loop converts each of these columns into category dtype. #Why is it necessary?
#Memory Efficiency: Categorical columns take less memory because pandas intern #Faster Computation: Operations like grouping or counting are faster on catego #Correct Semantics: Declaring a column as categorical tells pandas (and statis #Better Visualizations and Analysis: Some libraries (like seaborn or statsmode

cat_cols = ["Gender", "Primary_Device", "Exceeded_Recommended_Limit", "Health_
for c in cat_cols:
df[c] = df[c].astype("category")
 
#Quick Integrity Checks

print("Missing values by column:\n", df.isna().sum()) print("Duplicate rows:", df.duplicated().sum())
Missing values by column:
Age	0
Gender	0
Avg_Daily_Screen_Time_hr	0
Primary_Device	0
Exceeded_Recommended_Limit	0
Educational_to_Recreational_Ratio	0
Health_Impacts	3218
Urban_or_Rural	0
dtype: int64 Duplicate rows: 44
 
# Create Age Bands (Optional, for grouped analysis & ANOVA)

age_bins = [7, 10, 13, 16, 18]  # inclusive edges: (7-10], (10-13], (13-16], (
age_labels = ["8-10", "11-13", "14-16", "17-18"]
df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=Tr

# Explanation:
#age_bins defines the boundaries of each group (e.g., 7–10, 10–13). #age_labels assigns a human-readable label for each range.
#pd.cut() is used to categorize continuous values (ages) into discrete bins. #The result is a new column Age_Group that contains labels like "8-10", "11-13
# Why is it important?
#Simplifies Analysis: Instead of analyzing every single age (8, 9, 10, etc.), #Needed for ANOVA: Many statistical tests and visualizations compare groups, n #Improves Visualization: Bar charts and boxplots are easier to read when age i #Real-world Meaning: Age groups (like 8–10 years) often reflect developmental

#Descriptive Statistics:
#Numeric Summary
df.describe(include=[np.number]).T
	count	mean	std	min	25%	50%
Age	9712.0	12.979201	3.162437	8.0	10.00	13.00
Avg_Daily_Screen_Time_hr	9712.0	4.352837	1.718232	0.0	3.41	4.44
Educational_to_Recreational_Ratio	9712.0	0.427226	0.073221	0.3	0.37	0.43
#Explanation:
#df.describe(): Provides summary statistics (mean, min, max, quartiles, etc.) #include=[np.number]: Tells pandas to include only numeric columns (e.g., int, #T: Transposes the result so that rows become columns and vice versa — making



#Why is it important?
#Quick Data Overview: It displays the central tendency (mean), spread (standar #Spot Outliers or Errors: Large gaps between min/max and quartiles can hint at #Guided Analysis: You can identify which numeric features vary significantly o

for c in ["Gender","Primary_Device","Exceeded_Recommended_Limit","Health_Impac
if c in df.columns: display(pd.DataFrame(df[c].value_counts()).rename(columns={c:"counts"}
count
Gender

Male	4942


count

Primary_Device		
Smartphone	4568	
TV	2487	
Laptop	1433	
Tablet	1224	

count
Exceeded_Recommended_Limit

True	8301

 

Health_Impacts	count
Poor Sleep	2268
Poor Sleep, Eye Strain	979
Eye Strain	644
Poor Sleep, Anxiety	608
Poor Sleep, Obesity Risk	452
Anxiety	385
Poor Sleep, Eye Strain, Anxiety	258
Obesity Risk	252
Poor Sleep, Eye Strain, Obesity Risk	188
Eye Strain, Anxiety	135
Eye Strain, Obesity Risk	106
Poor Sleep, Anxiety, Obesity Risk	78
Anxiety, Obesity Risk	69
Poor Sleep, Eye Strain, Anxiety, Obesity Risk	37
Eye Strain, Anxiety, Obesity Risk	35

count
Urban_or_Rural

Urban	6851



Age_Group	count
8-10	2674
11-13	2643
14-16	2636
17-18	1759

 
#Why is it important?
#Data Distribution: It shows how balanced or imbalanced categories are (e.g., #Data Cleaning: Helps identify unexpected values (e.g., misspellings like "Fem

#Visualization & Modeling: Guides that categories might need grouping or balan
#Quick Key Metrics (Save for Report)

n= len(df)
mean_screen= df["Avg_Daily_Screen_Time_hr"].mean() std_screen= df["Avg_Daily_Screen_Time_hr"].std() prop_exceed = df["Exceeded_Recommended_Limit"].mean() print(f"Records:{n:,}")
print(f"mean daily screen time:{mean_screen:.2f} hr(SD={std_screen:.2f})\") print(f"%Exceeding recommended limit:{prop_exceed*100:.1f}%")
n= len(df)
mean_screen= df["Avg_Daily_Screen_Time_hr"].mean() std_screen= df["Avg_Daily_Screen_Time_hr"].std() prop_exceed = df["Exceeded_Recommended_Limit"].mean()
print(f"Records:{n:,}") # Removed the backslash before the closing quote print(f"mean daily screen time:{mean_screen:.2f} hr(SD={std_screen:.2f})") # print(f"%Exceeding recommended limit:{prop_exceed*100:.1f}%")

n = len(df)

# Convert categorical columns to numeric type
df["Avg_Daily_Screen_Time_hr"] = pd.to_numeric(df["Avg_Daily_Screen_Time_hr"], # If "Exceeded_Recommended_Limit" is also categorical and contains 0/1 or True if df["Exceeded_Recommended_Limit"].dtype.name == 'category':
df["Exceeded_Recommended_Limit"] = pd.to_numeric(df["Exceeded_Recommended_

# Now calculate statistics
mean_screen = df["Avg_Daily_Screen_Time_hr"].mean() std_screen = df["Avg_Daily_Screen_Time_hr"].std() prop_exceed = df["Exceeded_Recommended_Limit"].mean()

print(f"Records:{n:,}")
print(f"mean daily screen time:{mean_screen:.2f} hr(SD={std_screen:.2f})") print(f"%Exceeding recommended limit:{prop_exceed*100:.1f}%")
Records:9,712
mean daily screen time:4.35 hr(SD=1.72)
%Exceeding recommended limit:85.5%
#The "Quick Key Metrics" section calculates three essential summary statistics

Total Records (n) – n = len(df)
This tells you how many rows (children) are in the dataset. It's important for Mean and Standard Deviation of Screen Time

 mean_screen = df["Avg_Daily_Screen_Time_hr"].mean() std_screen = df["Avg_Daily_Screen_Time_hr"].std()
Mean shows the average daily screen time across all kids.

Standard Deviation (SD) tells how much screen time varies among kids.
Proportion Exceeding Recommended Limit python
Copy Edit
prop_exceed = df["Exceeded_Recommended_Limit"].mean()
Since True is treated as 1 and False as 0, the mean here gives the percentage

#ummarizing Data: These metrics give an immediate snapshot of the dataset's ce
For Reporting: They're easy to communicate (e.g., "The average screen time is Baseline Check: Helps verify if data is realistic and aligns with expected pat
 
#Visualization: Univariate Distributions #Screen Time Histogram & KDE
sns.histplot(df["Avg_Daily_Screen_Time_hr"], kde=True, bins=30) plt.xlabel("Avg Daily Screen Time (hours)") plt.title("Distribution of Daily Screen Time")
plt.show()
<img width="955" height="624" alt="image" src="https://github.com/user-attachments/assets/98f9de33-cef1-45fb-964e-76954514bf9d" />

#Boxplot by Age Group

sns.boxplot(x="Age_Group", y="Avg_Daily_Screen_Time_hr", data=df) plt.title("Screen Time by Age Group")
plt.show()
<img width="953" height="640" alt="image" src="https://github.com/user-attachments/assets/c00a0f7e-0849-4fac-8ffe-d4c37bd2c08a" />

#Count Plots for Categoricals

def plot_count(col):
sns.countplot(x=col, data=df, order=df[col].value_counts().index) plt.title(f"Counts:{col}")
plt.xticks(rotation=45) plt.show()

for col in ["Gender", "Primary_Device", "Exceeded_Recommended_Limit","Urban_or plot_count(col)
<img width="955" height="663" alt="image" src="https://github.com/user-attachments/assets/95aa3367-b42c-4bf3-a695-099ad358e414" />
<img width="943" height="696" alt="image" src="https://github.com/user-attachments/assets/7cabb69a-71c9-4793-996d-d001297dc048" />
<img width="955" height="654" alt="image" src="https://github.com/user-attachments/assets/7e0d76d6-9520-4aa5-9812-7f6b4a11b032" />
<img width="954" height="662" alt="image" src="https://github.com/user-attachments/assets/e20f07b5-98b7-4212-8faa-83ada029b1ae" />

# Bivariate Exploration
# Mean Screen Time by Category
(df.groupby("Age_Group")["Avg_Daily_Screen_Time_hr"].mean()
.sort_index()
.plot(kind="bar", ylabel="Mean Screen Time (hr)", title="Mean Screen Time b plt.show()
<img width="950" height="684" alt="image" src="https://github.com/user-attachments/assets/fbbfeefb-8f7c-419c-b501-2c0b1ef53c20" />

for col in ["Gender","Primary_Device","Urban_or_Rural","Exceeded_Recommended_L (df.groupby(col)["Avg_Daily_Screen_Time_hr"].mean()
.sort_values(ascending=False)
.plot(kind="bar", ylabel="Mean Screen Time (hr)", title=f"Mean Screen T plt.show()
<img width="950" height="693" alt="image" src="https://github.com/user-attachments/assets/4e28813d-eeb8-46b4-9555-aae187c75558" />

<img width="950" height="749" alt="image" src="https://github.com/user-attachments/assets/4f5e966b-8de0-4ce0-9d3d-7749b1542cd5" />
<img width="950" height="687" alt="image" src="https://github.com/user-attachments/assets/1f1b3287-6782-4a52-a552-980574305f8a" />
<img width="950" height="677" alt="image" src="https://github.com/user-attachments/assets/1ace9583-8c9a-453f-b153-b574f7992817" />
#Proportion Exceeding Limit by Group
def prop_exceed_by(col):
tmp = df.groupby(col)["Exceeded_Recommended_Limit"].mean().sort_values(asc tmp.plot(kind="bar", ylabel="Proportion > Limit", ylim=(0,1), title=f"% Ex plt.show()

for col in ["Age_Group","Gender","Primary_Device","Urban_or_Rural"]: prop_exceed_by(col)
<img width="958" height="677" alt="image" src="https://github.com/user-attachments/assets/8718271b-c4eb-4bf9-82f2-193c727c7e81" />
<img width="959" height="685" alt="image" src="https://github.com/user-attachments/assets/dc833384-3b61-4a5c-bd4d-8df40b9b1d39" />
<img width="959" height="741" alt="image" src="https://github.com/user-attachments/assets/792cbda6-5b4d-445e-a10b-80766eb8dad6" />
<img width="959" height="680" alt="image" src="https://github.com/user-attachments/assets/51a839de-b76f-46b7-98fa-74c693d771bb" />

# Relationship: Educational Ratio vs Screen Time

sns.scatterplot(x="Educational_to_Recreational_Ratio", y="Avg_Daily_Screen_Tim plt.title("Educational Ratio vs Screen Time")
plt.show()

# Correlation
corr = df[["Avg_Daily_Screen_Time_hr","Educational_to_Recreational_Ratio"]].co print(f"Correlation (Pearson r): {corr:.3f}")
<img width="953" height="640" alt="image" src="https://github.com/user-attachments/assets/eeab7443-8829-451d-becf-e0c32ed5fca3" />
Correlation (Pearson r): -0.088

#One-Way ANOVA: Screen Time ~ Age_Group

# Drop rows without Age_Group (should be none if binning succeeded)
ana_df = df.dropna(subset=["Age_Group"]).copy()
model_age = smf.ols('Avg_Daily_Screen_Time_hr ~ C(Age_Group)', data=ana_df).fi sm.stats.anova_lm(model_age, typ=2)
sum_sq	df	F	PR(>F)
C(Age_Group)	676.095600	3.0	78.15433	5.967154e-50
Residual  27993.910949	9708.0	NaN	NaN

# Check Assumptions
# Residual normality (QQ plot) sm.qqplot(model_age.resid, line='s') plt.title('QQ plot residuals: One-Way ANOVA') plt.show()

# Levene test for equal variances
lev = stats.levene(*[ana_df.loc[ana_df.Age_Group==lvl, "Avg_Daily_Screen_Time_ print("Levene test: stat=%.3f, p=%.3g" % (lev.statistic, lev.pvalue))
<img width="958" height="637" alt="image" src="https://github.com/user-attachments/assets/0e97ba4a-86eb-41a7-ad61-4a825b74c8ef" />
Levene test: stat=922.472, p=0

#Effect Size (Eta-Squared)
anova_res = sm.stats.anova_lm(model_age, typ=2) ss_between = anova_res.loc['C(Age_Group)', 'sum_sq']
ss_total	= ss_between + anova_res.loc['Residual','sum_sq'] eta_sq = ss_between/ss_total
print(f"Eta-squared: {eta_sq:.3f} (proportion of variance explained by Age_Gro
Eta-squared: 0.024 (proportion of variance explained by Age_Group)

#Post-Hoc: Tukey HSD Pairwise Comparisons

comp = mc.MultiComparison(ana_df['Avg_Daily_Screen_Time_hr'], ana_df['Age_Grou tukey_res = comp.tukeyhsd()
print(tukey_res.summary())
Multiple Comparison of Means - Tukey HSD, FWER=0.05
====================================================
group1	group2	meandiff	p-adj	lower	upper	reject
11-13	14-16	-0.0141	0.9904	-0.1342	0.106	False
11-13	17-18	0.0136	0.9939	-0.1207	0.1478	False
11-13	8-10	-0.5922	0.0	-0.7119	-0.4725	True
14-16	17-18	0.0277	0.9519	-0.1066	0.162	False
14-16	8-10	-0.5781	0.0	-0.6979	-0.4583	True
17-18	8-10	-0.6058	0.0	-0.7397	-0.4718	True

#Two-Way ANOVA: Screen Time ~ Age_Group * Gender
model_age_gender = smf.ols('Avg_Daily_Screen_Time_hr ~ C(Age_Group) * C(Gender sm.stats.anova_lm(model_age_gender, typ=2)

	sum_sq	df	F	PR(>F)
C(Age_Group)	677.233207	3.0	78.334366	4.591786e-50
C(Gender)	7.384919	1.0	2.562602	1.094509e-01
C(Age_Group):C(Gender)	21.485935	3.0	2.485240	5.877213e-02
Residual	27965.040095	9704.0	NaN	NaN


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Encode categoricals via one-hot (drop_first to avoid multicollinearity)
X = pd.get_dummies(df.drop(columns=["Exceeded_Recommended_Limit"]), drop_first y = df["Exceeded_Recommended_Limit"].cat.codes # True=1, False=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, strat

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
# Encode categoricals via one-hot (drop_first to avoid multicollinearity)
X = pd.get_dummies(df.drop(columns=["Exceeded_Recommended_Limit"]), drop_first

# Convert to category first if needed, or directly map True/False to 1/0 # Option 1: If column is boolean
y = df["Exceeded_Recommended_Limit"].astype(int)  # True=1, False=0

# Option 2: If you want to ensure it's categorical first
# y = df["Exceeded_Recommended_Limit"].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, strat



#Logistic Regression (Baseline)
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(max_iter=500, class_weight='balanced')  # imbalance
logit.fit(X_train, y_train)

pred = logit.predict(X_test)
proba = logit.predict_proba(X_test)[:,1]

print(classification_report(y_test, pred)) print("ROC-AUC:", roc_auc_score(y_test, proba))
precision	recall	f1-score	support
0	0.93	1.00	0.97	282
1	1.00	0.99	0.99	1661
accuracy		0.99	1943
macro avg	0.97	0.99	0.98	1943
weighted avg	0.99	0.99	0.99	1943
ROC-AUC: 0.999908198513243			

#Select & Scale Features

features = ["Age", "Avg_Daily_Screen_Time_hr", "Educational_to_Recreational_Ra
X = df[features].copy() scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Choose Number of Clusters (Elbow + Silhouette)
ks = range(2,9) inertias = [] sil_scores = [] for k in ks:
km = KMeans(n_clusters=k, random_state=42, n_init=10) labels = km.fit_predict(X_scaled) inertias.append(km.inertia_) sil_scores.append(silhouette_score(X_scaled, labels))

fig, ax1 = plt.subplots()

ax1.plot(ks, inertias, marker='o') ax1.set_xlabel('k (clusters)')
ax1.set_ylabel('Inertia (lower better)', color='tab:blue')

plt.twinx()
plt.plot(ks, sil_scores, marker='s') plt.ylabel('Silhouette Score (higher better)') plt.title('Elbow + Silhouette for K-Means') plt.show()

<img width="953" height="567" alt="image" src="https://github.com/user-attachments/assets/f8e2d5dc-c295-4d8c-b015-21f8f7f28caa" />

#Fit Final Model & Attach Cluster Labels
best_k = 4 # inspect plot; adjust
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10) df["Cluster"] = km_final.fit_predict(X_scaled)

#Profile Clusters

cluster_profile = (df.groupby("Cluster")
.agg(n=("Cluster","size"),
mean_age=("Age","mean"), mean_screen=("Avg_Daily_Screen_Time_hr","mean"), mean_edu_ratio=("Educational_to_Recreational_Ratio"," pct_exceed=("Exceeded_Recommended_Limit","mean"))
.sort_values("mean_screen", ascending=False))
cluster_profile

Cluster	n	mean_age	mean_screen	mean_edu_ratio	pct_exceed
2	1878	9.863685	5.958120	0.487167	1.000000
3	3337	14.150435	4.501510	0.349254	0.908301
1	2800	15.802857	4.494089	0.441064	0.912857
0	1697	9.464938	2.050919	0.491385	0.492634
					

#Visualize Clusters (2D Projections)
sns.scatterplot(x="Age", y="Avg_Daily_Screen_Time_hr", hue="Cluster", data=df, plt.title("Clusters by Age & Screen Time")
plt.show()

sns.scatterplot(x="Educational_to_Recreational_Ratio", y="Avg_Daily_Screen_Tim plt.title("Clusters by Educational Ratio & Screen Time")
plt.show()
<img width="953" height="638" alt="image" src="https://github.com/user-attachments/assets/28b59dc6-6e66-40ba-bd6f-11d1594b9ab1" />
<img width="953" height="640" alt="image" src="https://github.com/user-attachments/assets/6d10c17d-1cad-4d2b-9d30-84320b2d88a5" />

#Health Impact Exploration #Simplify to Any/None
df["Any_Health_Issue"] = (df["Health_Impacts"] != "None").astype(int)

(df.groupby("Any_Health_Issue")["Avg_Daily_Screen_Time_hr"].describe())


Any_Health_Issue
 
count	mean	std  min  25%  50%  75%	max
 

 
1  9712.0  4.352837  1.718232	0.0	3.41	4.44	5.38  13.89

#Bar Chart of Average Screen Time by Health Impact Category (Top 10) health_counts = df["Health_Impacts"].value_counts().head(10).index sns.barplot(y=health_counts, x=[df.loc[df["Health_Impacts"]==h, "Avg_Daily_Scr plt.xlabel("Mean Screen Time (hr)")
plt.ylabel("Health Impact")
plt.title("Avg Screen Time by Top Health Impact Categories") plt.show()
<img width="963" height="448" alt="image" src="https://github.com/user-attachments/assets/6ffb3020-c6df-48a5-866f-41fb852d1893" />


#Sample Interpreted Findings
#Mean daily screen time: ~4.35 hr (SD ~1.72).
Exceeding limit: ~85% of records flagged True (suggests the guideline threshol Age differences: Kids ages 8–10 average just under ~4 hr/day, while Ages 11+ a Gender: Mean screen time was very similar for males and females (~4.3–4.4 hr); Device: Smartphone most common; Laptops showed slightly higher mean usage than Educational ratio: Centered around ~0.43 with a weak negative correlation (r ≈



