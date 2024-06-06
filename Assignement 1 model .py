#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:08:16 2024

@author: hugoramos
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data from CSV files
clubs = pd.read_csv('clubs.csv')
players = pd.read_csv('players.csv')
competitions = pd.read_csv('competitions.csv')

# If needed, rename columns to ensure clear and consistent names for merging
clubs.rename(columns={'club_id': 'current_club_id', 'domestic_competition_id': 'competition_id'}, inplace=True)
clubs.drop(columns='last_season', inplace=True)

# Merge players with clubs
players_clubs_merged = pd.merge(players, clubs, on='current_club_id', how='inner')

# Merge the result with competitions
df = pd.merge(players_clubs_merged, competitions, on='competition_id', how='inner')

# Optionally rename columns for clarity
df.rename(columns={
    'name_x': 'player_name',
    'name_y': 'club_name',
    'url_x': 'player_url',
    'url_y': 'club_url'
}, inplace=True)

# Drop any unnecessary columns if needed
columns_to_drop = ['player_code', 'player_url', 'club_url', 'competition_code', 'type', 'confederation']
df.drop(columns=columns_to_drop, inplace=True)

# Combine first and last names into a single full name column and drop the original columns
df['full_name'] = df['first_name'] + ' ' + df['last_name']
df.drop(columns=['first_name', 'last_name'], inplace=True)

# Add 'Still in activity' column
df['Still in activity'] = df['last_season'].apply(lambda x: 'Yes' if x == 2023 else 'No')

# Add 'Has an agent?' column
df['Has an agent?'] = df['agent_name'].apply(lambda x: 'Yes' if pd.notnull(x) else 'No')

# Clean 'net_transfer_record' column
def convert_transfer_record(value):
    value = value.replace('€', '').replace('m', 'e6').replace('k', 'e3').replace('+', '').replace('-', '')
    try:
        return float(value)
    except ValueError:
        return np.nan

df['net_transfer_record'] = df['net_transfer_record'].apply(convert_transfer_record)

# Ensure all values in 'market_value_in_eur' are numeric
df['market_value_in_eur'] = pd.to_numeric(df['market_value_in_eur'], errors='coerce')

# Replace 'Unknown' in 'height_in_cm' and 'stadium_seats' with NaN
df['height_in_cm'].replace('Unknown', np.nan, inplace=True)
df['stadium_seats'].replace('Unknown', np.nan, inplace=True)

# Replace missing values in 'height_in_cm' and 'stadium_seats' with mean
df['height_in_cm'].fillna(df['height_in_cm'].mean(), inplace=True)
df['stadium_seats'].fillna(df['stadium_seats'].mean(), inplace=True)

for column in ['height_in_cm', 'stadium_seats']:
    df[column].fillna(df[column].mean(), inplace=True)

# Replace missing values in categorical columns with mode
for column in ['foot', 'position', 'club_name', 'country_name', 'competition_id']:
    df[column].fillna(df[column].mode()[0], inplace=True)
# Creating a column with the value of the players in Million
df['market_value_in_millions'] = df['market_value_in_eur'] / 1e6

# Filter the dataframe for market value above 1 million
df_filtered = df[df['market_value_in_millions'] >= 1]

# Drop 'market_value_in_eur' and 'highest_market_value_in_eur' columns
df_filtered.drop(['market_value_in_eur', 'highest_market_value_in_eur'], axis=1, inplace=True)

# Save the processed DataFrame to a CSV file
df_filtered.to_csv('processed_players_data.csv', index=False)

# Define the features and the target variable
X = df_filtered[['height_in_cm', 'foot', 'position', 'club_name', 'country_name', 'competition_id', 'stadium_seats']]
y = df_filtered['market_value_in_millions']

# Column transformer setup (including all relevant features)
column_transformer = ColumnTransformer(
    [
        ('num', StandardScaler(), ['height_in_cm', 'stadium_seats']),
        ('cat', OneHotEncoder(drop='first'), ['foot', 'position', 'club_name', 'country_name', 'competition_id'])
    ],
    remainder='passthrough'  # This will keep all other columns
)

# Transform the features
X_transformed = column_transformer.fit_transform(X)

print("Shape of transformed features:", X_transformed.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")

# Predict on the training set
y_train_pred = model.predict(X_train)

# Predict on the test set
y_test_pred = model.predict(X_test)

# Evaluate the model on the training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate the model on the test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training Set Evaluation:")
print("Mean Squared Error:", train_mse)
print("R-squared Score:", train_r2)

print("Testing Set Evaluation:")
print("Mean Squared Error:", test_mse)
print("R-squared Score:", test_r2)

# Save the model and the column transformer
joblib.dump(model, 'player_value_model.joblib')
joblib.dump(column_transformer, 'column_transformer.joblib')
