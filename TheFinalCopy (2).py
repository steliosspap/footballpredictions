#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load all the csv files provided for the competition 
group_round_games_df = pd.read_csv('NSL_Group_Round_Games.csv')

knockout_round_games_df = pd.read_csv('NSL_Knockout_Round_Games.csv')

metadata_df = pd.read_csv('NSL_Metadata.csv')

regular_season_data_df = pd.read_csv('NSL_regular_season_data_2.csv')





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


# We are going to calculate some statistics for each team based on their regular season performance
# In this case, Wins, Losses, and Draws for each team
regular_season_data_df['Winner'] = regular_season_data_df.apply(lambda x: x['HomeTeam'] if x['HomeScore'] > x['AwayScore'] else (x['AwayTeam'] if x['HomeScore'] < x['AwayScore'] else 'Draw'), axis=1)

# Calculate win, loss, draw counts, goals scored, goals conceded, and goal difference for each team
team_stats = {
    'Wins': regular_season_data_df.groupby('Winner').size(),
    'Goals Scored': regular_season_data_df.groupby('HomeTeam')['HomeScore'].sum() + regular_season_data_df.groupby('AwayTeam')['AwayScore'].sum(),
    'Goals Conceded': regular_season_data_df.groupby('HomeTeam')['AwayScore'].sum() + regular_season_data_df.groupby('AwayTeam')['HomeScore'].sum(),
}

team_stats_df = pd.DataFrame(team_stats).fillna(0)  # Fill NA values with 0 for teams without wins
team_stats_df['Goal Difference'] = team_stats_df['Goals Scored'] - team_stats_df['Goals Conceded']
team_stats_df = team_stats_df.drop('Draw', errors='ignore')  # Remove the 'Draw' row if exists

# Calculate win rate for each team
total_games = regular_season_data_df.groupby('HomeTeam').size() + regular_season_data_df.groupby('AwayTeam').size()
team_stats_df['Win Rate'] = team_stats_df['Wins'] / total_games

# Calculate expected goals (xG) metrics
team_stats_df['xG Scored'] = regular_season_data_df.groupby('HomeTeam')['Home_xG'].sum() + regular_season_data_df.groupby('AwayTeam')['Away_xG'].sum()
team_stats_df['xG Conceded'] = regular_season_data_df.groupby('HomeTeam')['Away_xG'].sum() + regular_season_data_df.groupby('AwayTeam')['Home_xG'].sum()
team_stats_df['xG Difference'] = team_stats_df['xG Scored'] - team_stats_df['xG Conceded']







# In[14]:


# There, we will calculate home and away specific statistics
home_metrics = regular_season_data_df.groupby('HomeTeam').agg({
    'HomeScore': ['sum', 'count'],
    'AwayScore': 'sum',
    'Home_xG': 'sum',
    'Away_xG': 'sum'
}).rename(columns={'sum': 'Total', 'count': 'Games'})

away_metrics = regular_season_data_df.groupby('AwayTeam').agg({
    'AwayScore': ['sum'],
    'HomeScore': 'sum',
    'Away_xG': 'sum',
    'Home_xG': 'sum'
}).rename(columns={'sum': 'Total'})

# Rename columns for clarity
home_metrics.columns = ['HomeGoalsScored', 'HomeGames', 'HomeGoalsConceded', 'Home_xG', 'Home_xG_Conceded']
away_metrics.columns = ['AwayGoalsScored', 'AwayGoalsConceded', 'Away_xG', 'Away_xG_Conceded']

# Calculate home and away win rates and other metrics
home_metrics['HomeWinRate'] = home_metrics.apply(lambda x: (x['HomeGoalsScored'] > x['HomeGoalsConceded']).mean(), axis=1)
away_metrics['AwayWinRate'] = away_metrics.apply(lambda x: (x['AwayGoalsScored'] > x['AwayGoalsConceded']).mean(), axis=1)

# Combine home and away metrics into a single dataframe
team_performance_df = home_metrics.join(away_metrics, how='outer')

# Calculate overall metrics
team_performance_df['TotalGoalsScored'] = team_performance_df['HomeGoalsScored'] + team_performance_df['AwayGoalsScored']
team_performance_df['TotalGoalsConceded'] = team_performance_df['HomeGoalsConceded'] + team_performance_df['AwayGoalsConceded']
team_performance_df['Total_xG'] = team_performance_df['Home_xG'] + team_performance_df['Away_xG']
team_performance_df['Total_xG_Conceded'] = team_performance_df['Home_xG_Conceded'] + team_performance_df['Away_xG_Conceded']
team_performance_df['OverallWinRate'] = (team_performance_df['HomeWinRate'] + team_performance_df['AwayWinRate']) / 2

# Drop intermediate columns to clean up the dataframe
team_performance_clean_df = team_performance_df.drop(columns=[
    'HomeWinRate', 'AwayWinRate'
])

# Reset index to have Team as a column
team_performance_clean_df.reset_index(inplace=True)
team_performance_clean_df.rename(columns={'index': 'Team'}, inplace=True)




# In[15]:


scaler = StandardScaler()

columns_to_standardize = [
    'HomeGoalsScored', 'HomeGoalsConceded', 'Home_xG', 'Home_xG_Conceded',
    'AwayGoalsScored', 'AwayGoalsConceded', 'Away_xG', 'Away_xG_Conceded',
    'TotalGoalsScored', 'TotalGoalsConceded', 'Total_xG', 'Total_xG_Conceded'
]

# Apply standardization
team_performance_clean_df[columns_to_standardize] = scaler.fit_transform(team_performance_clean_df[columns_to_standardize])



# In[21]:


# At this moment we decided to add more features to our system, so this is the final step before extracting all our standardized data into a single csv file
# We will additionally calculate the league averages for goals scored, goals conceded, and xG metrics
league_averages = {
    'avg_goals_scored': team_performance_clean_df['TotalGoalsScored'].mean(),
    'avg_goals_conceded': team_performance_clean_df['TotalGoalsConceded'].mean(),
    'avg_xG_scored': team_performance_clean_df['Total_xG'].mean(),
    'avg_xG_conceded': team_performance_clean_df['Total_xG_Conceded'].mean(),
}

# Calculate Attack Strength, Defense Strength, and Home/Away Performance Differential
team_performance_clean_df['AttackStrength'] = team_performance_clean_df['TotalGoalsScored'] / league_averages['avg_goals_scored']
team_performance_clean_df['DefenseStrength'] = team_performance_clean_df['TotalGoalsConceded'] / league_averages['avg_goals_conceded']
team_performance_clean_df['HomeAdvantage'] = (team_performance_clean_df['HomeGoalsScored'] - team_performance_clean_df['HomeGoalsConceded']) - (team_performance_clean_df['AwayGoalsScored'] - team_performance_clean_df['AwayGoalsConceded'])

# Normalize the newly created features
new_features_to_normalize = ['AttackStrength', 'DefenseStrength', 'HomeAdvantage']
team_performance_clean_df[new_features_to_normalize] = scaler.fit_transform(team_performance_clean_df[new_features_to_normalize])


# Export the standardized team performance data to a CSV file
standardized_team_performance_csv_path = 'standardized_team_performance_metrics.csv'
team_performance_clean_df.to_csv(standardized_team_performance_csv_path, index=False)




# In[17]:


# These are the metrics that we have chosen to insert to our model 
# Create feature vectors for matches by calculating the difference in team metrics
# The target variable is the "HomeWin"
features_to_use = ['AttackStrength', 'DefenseStrength', 'HomeAdvantage']
match_features = pd.DataFrame()

for feature in features_to_use:
    home_feature = regular_season_data_df['HomeTeam'].map(team_performance_clean_df.set_index('HomeTeam')[feature])
    away_feature = regular_season_data_df['AwayTeam'].map(team_performance_clean_df.set_index('HomeTeam')[feature])
    match_features[feature + '_diff'] = home_feature - away_feature

# Define the outcome variable: 1 for home win, 0 otherwise
match_features['HomeWin'] = (regular_season_data_df['HomeScore'] > regular_season_data_df['AwayScore']).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(match_features.drop('HomeWin', axis=1), match_features['HomeWin'], test_size=0.3, random_state=42)

# Output the shapes of the splits to confirm
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[18]:


# Initialize the logistic regression model
logreg = LogisticRegression(random_state=42)

# Train the model on the training data
logreg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]  # Probabilities of the positive class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# We will output the conclusions
accuracy, roc_auc


# In[19]:


#This is the part of our code where we manually insert the match names and we derive the desired outcome

# Specific matches to predict
matches_to_predict = pd.DataFrame({
    'HomeTeam': ['OAK', 'ALB'],
    'AwayTeam': ['PRO', 'CHM']
})

# Prepare the feature vectors for the matches
for feature in ['AttackStrength', 'DefenseStrength', 'HomeAdvantage']:
    matches_to_predict[f'{feature}_diff'] = (
        matches_to_predict['HomeTeam'].map(team_performance_clean_df.set_index('HomeTeam')[feature]) -
        matches_to_predict['AwayTeam'].map(team_performance_clean_df.set_index('HomeTeam')[feature])
    )

# Use the logistic regression model to predict outcomes
predicted_outcomes = logreg.predict(matches_to_predict[['AttackStrength_diff', 'DefenseStrength_diff', 'HomeAdvantage_diff']])
predicted_probabilities = logreg.predict_proba(matches_to_predict[['AttackStrength_diff', 'DefenseStrength_diff', 'HomeAdvantage_diff']])

# Add predictions to the DataFrame
matches_to_predict['PredictedOutcome'] = predicted_outcomes
matches_to_predict['WinProbability'] = predicted_probabilities[:, 1]  # Probability of the home team winning

matches_to_predict[['HomeTeam', 'AwayTeam', 'PredictedOutcome', 'WinProbability']]




# In[ ]:





# In[ ]:




