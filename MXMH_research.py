import msn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import missingno as msn
file_path = "C:/Users/15154/Desktop/mxmh_survey_results.csv"
df = pd.read_csv(file_path)
print(df.head())
df.info()
plt.figure(figsize=(16, 12))
msn.bar(df)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
plt.show()
print(df['Age'].describe())
df['High_Depression'] = (df['Depression'] > 6).astype(int)
df['High_Anxiety']=(df['Anxiety'] > 5).astype(int)
# age distribution
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

colors = sns.color_palette("coolwarm", 30)
axs[0].hist(df['Age'], bins=30, color=colors[-10], edgecolor=colors[0])
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Age Distribution')

sns.boxplot(df['Age'], color='#98df8a', ax=axs[1])
axs[1].set_title('Boxplot of Age Distribution')
axs[1].set_xlabel('Age')

plt.tight_layout()
plt.show()
# Count the number of age occurrences
age_counts = df['Age'].value_counts()
# age values whose occurrence times are greater than 40
filtered_ages = age_counts[age_counts > 40]
print(filtered_ages)
print(df['Age'].max())
print(df['Hours per day'].describe())
#hours per day distribution
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

sns.histplot(df['Hours per day'], bins=30, kde=True, color='#3b4cc0', ax=axs[0])
axs[0].set_title('Hours per Day Listening to Music')
axs[0].set_xlabel('Hours per Day')
axs[0].set_ylabel('Frequency')

sns.boxplot(df['Hours per day'], color='#ffb6c1', ax=axs[1])
axs[1].set_title('Boxplot of Hours per Day Listening to Music')
axs[1].set_xlabel('Hours per Day')

plt.tight_layout()
plt.show()
#hours per day vs age
#scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Hours per day', data=df, color='skyblue')
plt.title('Age vs Hours per Day Listening to Music')
plt.xlabel('Age')
plt.ylabel('Hours per Day')
plt.show()
#boxplot
age_bins = [10, 30, 50, 70, 90]
age_labels = ['10-29', '30-49', '50-69', '70-89']
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
plt.figure(figsize=(10, 6))
sns.boxplot(x='age_group', y='Hours per day', data=df, palette='Set2')
plt.title('Hours per Day Listening to Music by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Hours per Day')
plt.show()
#pie chart of PSS
colors = sns.color_palette("Set3")
services = df['Primary streaming service'].value_counts()
services.plot(kind='pie',autopct='%1.1f%%',startangle=140,shadow=True,colors=colors)
plt.title('Streaming services by popularity')
plt.ylabel("")
plt.show()
df['Primary streaming service'].value_counts(normalize=True)
# hours per day grouped by PSS
hours_per_service = df.groupby('Primary streaming service')['Hours per day'].mean()
print(hours_per_service)
hours_per_service_df = pd.DataFrame(hours_per_service).reset_index()
hours_per_service_df.replace(['Other streaming service', 'I do not use a streaming service.', 'YouTube Music'],
                       ['Other', 'None', 'YouTube'], inplace=True)
# bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Primary streaming service', y='Hours per day', data=hours_per_service_df, palette='Set2')
plt.title('Average Hours per Day by Streaming Service')
plt.xlabel('Streaming Service')
plt.ylabel('Average Hours per Day')
plt.xticks(fontsize=7)
plt.show()
# description statistics of age groups
age_stats = df.groupby('Primary streaming service')['Age'].describe()
print(age_stats)
#Boxplot of age distribution of different pss
df.replace(['Other streaming service', 'I do not use a streaming service.', 'YouTube Music'],
                       ['Other', 'None', 'YouTube'], inplace=True)
bplot = sns.boxplot(data=df, x="Primary streaming service", y = "Age", showfliers = False)
plt.title('Streaming services by Age')
#Pie chart of users using different pss by age group
age_bins = [10, 30, 50, 70, 90]
age_labels = ['10-29', '30-49', '50-69', '70-89']
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, age_group in zip(axes.flatten(), age_labels):
    group_data = df[df['age_group'] == age_group]['Primary streaming service'].value_counts()
    group_data.plot(kind='pie', autopct='%1.1f%%', startangle=140, shadow=True, ax=ax)
    ax.set_title(f'Streaming Service Distribution for Age Group {age_group}')
    ax.set_ylabel("")
plt.tight_layout()
plt.show()
# Instrumentalist vs Composer distribution
instrumentalist_counts = df['Instrumentalist'].value_counts()
composer_counts = df['Composer'].value_counts()
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
instrumentalist_counts.plot(kind='pie', autopct='%1.1f%%',
                            colors=['lightblue', 'lightgreen'],
                            textprops={'fontsize': 20, 'color': 'orange', 'weight': 'bold'})
plt.title('Instrumentalist Distribution',fontsize=25)
plt.ylabel('')
plt.subplot(1, 2, 2)
composer_counts.plot(kind='pie', autopct='%1.1f%%',
                     colors=['lightcoral', 'lightgoldenrodyellow'],
                     textprops={'fontsize': 20, 'color': 'brown', 'weight': 'bold'})
plt.title('Composer Distribution',fontsize=25)
plt.ylabel('')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

df_encoded = pd.get_dummies(df, columns=['Instrumentalist', 'Composer', 'While working'])
relevant_columns = ['High_Anxiety', 'High_Depression'] + [col for col in df_encoded.columns if 'Instrumentalist' in col or 'Composer' in col or 'While working' in col]
corr_matrix = df_encoded[relevant_columns].corr()

plt.figure(figsize=(18,15))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix for Categorical Variables and High Anxiety/Depression')
plt.show()

# discribe mental health issues
mental_health_stats = df[['Anxiety', 'Depression', 'Insomnia', 'OCD']].describe()
print(mental_health_stats)
#bar plot and kde on
plt.figure(figsize=(14, 10))
for i, column in enumerate(['Anxiety', 'Depression', 'Insomnia', 'OCD'], 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[column], bins=25, kde=True, color='lightpink')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
for i, column in enumerate(['Anxiety', 'Depression', 'Insomnia', 'OCD'], 1):
    plt.subplot(2, 2, i)
    sns.regplot(x='Age', y=column, data=df, scatter_kws={'color':'skyblue'}, line_kws={'color':'red'})
    plt.title(f'Age vs {column}')
    plt.xlabel('Age')
    plt.ylabel(column)
plt.tight_layout()
plt.show()
#heatmap
selected_columns = ['Age', 'Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages']
df_encoded = df[selected_columns].copy()
for column in ['While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages']:
    df_encoded[column] = df_encoded[column].map({'Yes': 1, 'No': 0})#为非数值型数据编码
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
#hours per day of high mental issue scores
avg_hours_per_day = {}
for column in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
    avg_hours_per_day[column] = df[df[column] > 8]['Hours per day'].mean()
avg_hours_df = pd.DataFrame(list(avg_hours_per_day.items()), columns=['Mental Issue', 'Average Hours per Day'])
plt.figure(figsize=(8, 5))
sns.barplot(x='Mental Issue', y='Average Hours per Day', data=avg_hours_df, palette='viridis',width=0.4)
plt.title('Listening Hours for High Mental Issue Scores (>8)')
plt.xlabel('Mental Issue')
plt.ylabel('Average Hours per Day')
plt.ylim(3,6)
plt.xticks(fontsize=10)
plt.tight_layout()
plt.show()

#test
avg_hours_per_day1 = {}
for column in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
    avg_hours_per_day1[column] = df[df[column] >= 8]['Hours per day'].mean()
avg_hours_df1 = pd.DataFrame(list(avg_hours_per_day1.items()), columns=['Mental Issue', 'Average Hours per Day'])
plt.figure(figsize=(8, 5))
sns.barplot(x='Mental Issue', y='Average Hours per Day', data=avg_hours_df1, palette='viridis',width=0.4)
plt.title('Listening Hours for High Mental Issue Scores (>=8)')
plt.xlabel('Mental Issue')
plt.ylabel('Average Hours per Day')
plt.ylim(3,6)
plt.xticks(fontsize=10)
plt.tight_layout()
plt.show()

#hours per day of low mental issues scores
avg_hours_per_day2 = {}
for column in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
    avg_hours_per_day2[column] = df[df[column]< 3]['Hours per day'].mean()
avg_hours_df2 = pd.DataFrame(list(avg_hours_per_day2.items()), columns=['Mental Issue', 'Average Hours per Day'])
plt.figure(figsize=(8, 5))
sns.barplot(x='Mental Issue', y='Average Hours per Day', data=avg_hours_df2, palette='viridis',width=0.4)
plt.title('Listening Hours for Low Mental Issue Scores (<3)')
plt.xlabel('Mental Issue')
plt.ylabel('Average Hours per Day')
plt.ylim(3,4)
plt.xticks(fontsize=10)
plt.tight_layout()
plt.show()
#relationships between mental health issues
mental_health_df = df[['Anxiety', 'Depression', 'Insomnia', 'OCD']]
correlation_matrix = mental_health_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Mental Health Issues')
plt.show()
#music effects on mental health
effects=df['Music effects'].value_counts()
print(effects)
plt.figure(figsize=(5,5))
plt.title('Music Effects on Mental Health')
effects.plot(kind='pie',ylabel='')
plt.show()

# distribution of fav genre
df.replace(['Video game music'],['VGM'], inplace=True)
genre = df["Fav genre"].value_counts()
print(genre)
filtered_genre = genre[genre > 10]
plt.figure(figsize=(10, 8))
plt.pie(filtered_genre, labels=filtered_genre.index, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Top Genre Breakdown')
plt.ylabel("")
plt.show()

#age vs fav genre
sns.scatterplot(data=df,y="Fav genre",x="Age",alpha=0.5)
plt.title('Age Distribution by genre')
plt.show()
#music effects of different genres
all_genre=np.sort(df['Fav genre'].unique())
gp_df = df.groupby(['Fav genre'])
gp_count = gp_df['Music effects'].value_counts(ascending=False, normalize=True).unstack().fillna(0)
imp_col = gp_count['Improve']
none_col = gp_count['No effect']
wor_col = gp_count['Worsen']
width = 0.22
x = np.arange(len(all_genre))
fig, ax = plt.subplots(figsize=(13, 9))
b1 = ax.bar(x-width, imp_col, width, label="Improve", color = 'deepskyblue')
b2 = ax.bar(x, none_col, width, label="No effect", color = 'lightblue')
b3 = ax.bar(x+width, wor_col, width, label="Worsen", color = 'slateblue')
plt.title("Music effects by Favorite Genre")
ax.set_ylabel('Distribution')
ax.set_xlabel('Genre')
ax.set_xticks(x, all_genre, rotation = 45)
ax.legend()
plt.show()
#BPM
print(df['BPM'].max())
print(df['BPM'].min())
df = df[(df.BPM < 500) & (df.BPM > 20)]
bpm_stats = df['BPM'].describe()
print(bpm_stats)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['BPM'], bins=30, kde=True, color='blue')
plt.title('BPM Distribution (Histogram)')
plt.subplot(1, 2, 2)
sns.kdeplot(df['BPM'], shade=True, color='blue')
plt.title('BPM Distribution (Density Plot)')
plt.tight_layout()
plt.show()
#BPM vs age
plt.figure(figsize=(10, 6))
sns.boxplot(x='age_group', y='BPM', data=df, palette='Set3')
plt.title('BPM Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('BPM')
plt.show()
#BPM vs fav genre
plt.figure(figsize=(14, 8))
sns.boxplot(x='Fav genre', y='BPM', data=df, palette='Set3')
plt.xticks(rotation=45)
plt.title('BPM Distribution by Music Genre')
plt.xlabel('Music Genre')
plt.ylabel('BPM')
plt.show()
#heatmap of bpm and mental health
fig = plt.figure(figsize=(8, 5))
plt.suptitle("BPM vs Mental Health")
y = df["Anxiety"]
y2 = df["Depression"]
y3 = df["Insomnia"]
y4 = df["OCD"]
x = df["BPM"]

ax = fig.add_subplot(221)
plt.title('Anxiety')
plt.xticks([])
plt.ylabel('Mental health ranking')
plt.hist2d(x,y, density = True,cmap='inferno');

ax = fig.add_subplot(222)
plt.title('Depression')
plt.xticks([])
plt.hist2d(x,y2, density = True,cmap='inferno');

ax = fig.add_subplot(223)
plt.title('Insomnia')
plt.ylabel('Mental health ranking')
plt.xlabel('BPM')
plt.hist2d(x,y3, density = True,cmap='inferno');

ax = fig.add_subplot(224)
plt.title('OCD')
plt.xlabel('BPM')
plt.hist2d(x,y4, density = True,cmap='inferno')
plt.show()

#bpm 100-150
labelEncoder = LabelEncoder()
df["BPM 100-150"] = labelEncoder.fit_transform((df["BPM"] >= 100) & (df["BPM"] <= 150)).astype(int)

def calculate_means(dataset, pathology_name):
    means_in_range = dataset[dataset["BPM 100-150"] == 1][pathology_name].mean()
    means_out_of_range = dataset[dataset["BPM 100-150"] == 0][pathology_name].mean()

    return means_in_range, means_out_of_range

pathologies = ["Anxiety", "Depression"]
mean_values = {}

for pathology in pathologies:
    mean_values[pathology] = calculate_means(df, pathology)

data = {
    'Pathology': [],
    'Mean Score': [],
    'BPM Range': []
}

for pathology, (in_range, out_of_range) in mean_values.items():
    data['Pathology'].extend([pathology, pathology])
    data['Mean Score'].extend([in_range, out_of_range])
    data['BPM Range'].extend(['100-150', 'Out of 100-150'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Pathology', y='Mean Score', hue='BPM Range', data=data, palette='viridis')
plt.title('Mean Pathology Scores for Different BPM Ranges')
plt.xlabel('Pathology')
plt.ylabel('Mean Score')
plt.ylim(0, max(data['Mean Score']) + 1)
plt.show()
#The results showed that anxiety, depression, insomnia and OCD had no relationship with the BPM of the music the subjects listened to, and the average score of each mental state was similar under the two BPM categories.
#Distribution of listening frequencies by genre
plt.figure(figsize=(20, 16))
plt.suptitle("Frequency Distribution by Genre")
updated_genres = ['Classical', 'Country', 'EDM', 'Folk', 'Gospel', 'Hip hop', 'Jazz', 'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B', 'Rap', 'Rock', 'Video game music']
colors = ["#D4A017", "#F28E2B", "#E15759", "#76B7B2"]

for i, genre in enumerate(updated_genres):
    plt.subplot(5, 4, i + 1)
    sns.countplot(data=df, x=f'Frequency [{genre}]', order=df[f'Frequency [{genre}]'].value_counts().index, palette=colors)
    plt.title(f'Frequency of Listening to {genre}')
    plt.xlabel('')
    plt.ylabel('Count')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#The relationship between high depression levels and other variables
df_ax = df.drop([
    "Timestamp", "Primary streaming service", "Exploratory", "Foreign languages", "Anxiety", "Depression",
    "Insomnia", "OCD", "Music effects", "Permissions","age_group"
], axis=1)

frequency_mapping = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Very frequently": 3
}

frequency_columns = [
    "Frequency [Classical]", "Frequency [Country]", "Frequency [EDM]", "Frequency [Folk]",
    "Frequency [Gospel]", "Frequency [Hip hop]", "Frequency [Jazz]", "Frequency [K pop]",
    "Frequency [Latin]", "Frequency [Lofi]", "Frequency [Metal]", "Frequency [Pop]",
    "Frequency [R&B]", "Frequency [Rap]", "Frequency [Rock]", "Frequency [Video game music]"
]

for col in frequency_columns:
    df_ax[col] = df_ax[col].map(frequency_mapping)
df_ax['While working'] = df_ax['While working'].replace({'Yes': 1, 'No': 0})
df_ax['Instrumentalist'] = df_ax['Instrumentalist'].replace({'Yes': 1, 'No': 0})
df_ax['Composer'] = df_ax['Composer'].replace({'Yes': 1, 'No': 0})
label_encoder=LabelEncoder()
df_ax['Fav genre'] = label_encoder.fit_transform(df_ax['Fav genre'])
print(df_ax.info())
corr_matrix = df_ax.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma', cbar=True, linewidths=.5, annot_kws={"size": 12})
plt.title('Correlation Matrix with High Depression')
plt.show()
#Genre frequency and mental health scores
genres = ['Classical', 'Country', 'EDM', 'Folk', 'Gospel', 'Hip hop', 'Jazz', 'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B', 'Rap', 'Rock', 'Video game music']
mental_health_issues = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
frequency_mapping = ['Never', 'Rarely', 'Sometimes', 'Very frequently']

data_list = []

for genre in genres:
    for freq in frequency_mapping:
        avg_score = df[df[f'Frequency [{genre}]'] == freq][mental_health_issues].mean()
        data_list.append({'Genre': genre, 'Frequency': freq, 'Anxiety': avg_score['Anxiety'], 'Depression': avg_score['Depression'], 'Insomnia': avg_score['Insomnia'], 'OCD': avg_score['OCD']})

avg_scores = pd.DataFrame(data_list)

print(avg_scores.head())
fig, axes = plt.subplots(4, 1, figsize=(20, 30), sharex=True)

for idx, issue in enumerate(mental_health_issues):
    ax = axes[idx]
    sns.barplot(data=avg_scores, x='Genre', y=issue, hue='Frequency', ax=ax, palette='viridis')
    ax.set_title(f'{issue} Score by Genre and Listening Frequency')
    ax.set_ylabel(f'{issue} Score')
    ax.legend(title='Listening Frequency', loc='upper right')

plt.xlabel('Music Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#The correlation between genre frequency and Music effects
frequency_mapping = ['Never', 'Rarely', 'Sometimes', 'Very frequently']
effect_mapping = {'No effect': 0, 'Worsen': -1, 'Improve': 1}
df['Music effects'] = df['Music effects'].map(effect_mapping)
correlation_data = pd.DataFrame()

for genre in genres:
    genre_freq = df[f'Frequency [{genre}]'].map({'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3})
    correlation_data[genre] = genre_freq

correlation_data['Music effects'] = df['Music effects']
correlation_matrix = correlation_data.corr()
print(correlation_matrix)

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between Listening Frequency and Music Effects')
plt.show()