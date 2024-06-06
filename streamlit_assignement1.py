#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:01:13 2024

@author: hugoramos
"""
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



st.set_page_config(page_title="Player Recruitment App", layout="wide", page_icon="⚽")

# Custom CSS for styling (the background coding did not work, I used a different solution)
st.markdown(
    """
    <style>
    .css-1d391kg {
        background: linear-gradient(90deg, rgba(255, 255, 255, 1) 0%, rgba(240, 242, 246, 1) 100%);
    }
    .css-1cpxqw2, .css-1oe6wy4 {
        color: #262730;
    }
    .css-184tjsw p {
        color: #262730;
    }
    .stButton>button {
        color: white;
        background-color: #F63366;
        border: none;
        padding: 10px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 8px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #D32F2F;
        color: white;
    }
    .css-1q8dd3e p {
        color: #262730;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#loading the data set and model
def load_data_and_model():
    data = pd.read_csv('processed_players_data.csv')  
    model = joblib.load('player_value_model.joblib')
    column_transformer = joblib.load('column_transformer.joblib')
    return data, model, column_transformer

df, model, column_transformer = load_data_and_model()

#use the model 
def predict_market_value(player_details):
    features = pd.DataFrame({
        'height_in_cm': [player_details['height_in_cm']],
        'foot': [player_details['foot']],
        'position': [player_details['position']],
        'club_name': [player_details['club_name']],
        'country_name': [player_details['country_name']],
        'competition_id': [player_details['competition_id']],
        'stadium_seats': [player_details['stadium_seats']],  # Used for model but not displayed
        'net_transfer_record': [player_details['net_transfer_record']]
    })
    transformed_features = column_transformer.transform(features)
    prediction = model.predict(transformed_features)
    return prediction[0]



#streamlit app settings
st.sidebar.title("Football Player Agenda")
page = st.sidebar.radio("Go to", ["Player Recruitment", "Specific Player Search"])

if page == "Player Recruitment":
    st.markdown("<h1 style='text-align: center; color: #77C8C8; font-size: 50px;'>Player Recruitment App</h1>", unsafe_allow_html=True)
    st.sidebar.header('Choose the Settings')

    #sidebar search 
    position = st.sidebar.selectbox('Position', options=['All'] + df['position'].unique().tolist(), index=0)
    country_of_citizenship = st.sidebar.selectbox('Country Player is From', options=['All'] + df['country_of_citizenship'].unique().tolist(), index=0)
    country_name = st.sidebar.selectbox('Country Player is Playing In', options=['All'] + df['country_name'].unique().tolist(), index=0)
    club = st.sidebar.selectbox('Club', options=['All'] + df['club_name'].unique().tolist(), index=0)
    foot = st.sidebar.selectbox('Foot', options=['All'] + df['foot'].unique().tolist(), index=0)
    agent_status = st.sidebar.radio('Has an agent?', ['All', 'Yes', 'No'])
    height_min = st.sidebar.slider('Minimum Height (cm)', min_value=int(df['height_in_cm'].min()), max_value=int(df['height_in_cm'].max()), value=int(df['height_in_cm'].min()))
    height_max = st.sidebar.slider('Maximum Height (cm)', min_value=int(df['height_in_cm'].min()), max_value=int(df['height_in_cm'].max()), value=int(df['height_in_cm'].max()))

    #filters settings
    filtered_players = df[
        ((df['position'] == position) | (position == 'All')) &
        ((df['country_of_citizenship'] == country_of_citizenship) | (country_of_citizenship == 'All')) &
        ((df['country_name'] == country_name) | (country_name == 'All')) &
        ((df['club_name'] == club) | (club == 'All')) &
        ((df['foot'] == foot) | (foot == 'All')) &
        ((df['Has an agent?'] == agent_status) | (agent_status == 'All')) &
        (df['height_in_cm'] >= height_min) &
        (df['height_in_cm'] <= height_max) &
        (df['Still in activity'] == 'Yes')  # Filter players who are still in activity
    ]

    filtered_players = filtered_players.head(5)

    st.write(f"Showing {len(filtered_players)} players based on the selected filters:")

    for idx, player in filtered_players.iterrows():
        with st.expander(f"{player['player_name']} - {player['club_name']}"):
            if not pd.isnull(player['image_url']):
                st.image(player['image_url'], width=150)
            cols = st.columns([1, 1])
            cols[0].write('**Position:**')
            cols[1].write(player['position'])
            cols[0].write('**Foot:**')
            cols[1].write(player['foot'])
            cols[0].write('**Height (cm):**')
            cols[1].write(f"<div class='highlight-number'>{player['height_in_cm']}</div>", unsafe_allow_html=True)
            cols[0].write('**Country Player is From:**')
            cols[1].write(player['country_of_citizenship'])
            cols[0].write('**Country Player is Playing In:**')
            cols[1].write(player['country_name'])
            cols[0].write('**Net Transfer Record (Millions €):**')
            net_transfer_record_million = player['net_transfer_record'] / 1e6 if player['net_transfer_record'] else 'Unknown'
            cols[1].write(f"<div class='highlight-number'>{net_transfer_record_million}</div>", unsafe_allow_html=True)
            cols[0].write('**Still in activity:**')
            cols[1].write(player['Still in activity'])
            cols[0].write('**Has an agent?:**')
            cols[1].write(player['Has an agent?'])

            
            if st.button(f"Predict Market Value for {player['player_name']}", key=idx):
                estimated_value = predict_market_value(player)
                st.write(f"<div class='highlight-number'>Estimated Market Value (Millions €): {round(estimated_value, 2)}</div>", unsafe_allow_html=True)

    
    st.sidebar.header('About')
    st.sidebar.write('This app helps football agents to search and filter football players based on various criteria and predict their market value, it is a very usefull tool to find new clients for agent.')

elif page == "Specific Player Search":
    st.markdown("<h1 style='text-align: center; color: #77C8C8; font-size: 50px;'>Specific Player Search</h1>", unsafe_allow_html=True)

    #2nd page search for specific player
    player_name = st.text_input('Enter player name to search', '')

    if player_name:
        player = df[(df['player_name'].str.contains(player_name, case=False, na=False)) & (df['Still in activity'] == 'Yes')]

        if not player.empty:
            player = player.iloc[0]
            st.write(f"**Name:** {player['player_name']}")
            if not pd.isnull(player['image_url']):
                st.image(player['image_url'], width=150)
            st.write(f"**Position:** {player['position']}")
            st.write(f"**Foot:** {player['foot']}")
            st.write(f"**Height (cm):** {player['height_in_cm']}")
            st.write(f"**Country Player is From:** {player['country_of_citizenship']}")
            st.write(f"**Country Player is Playing In:** {player['country_name']}")
            st.write(f"**Still in activity:** {player['Still in activity']}")
            st.write(f"**Has an agent?:** {player['Has an agent?']}")

            #Graphs for similar player on market value
            height_range = (-5, 5) if player['height_in_cm'] >= 170 else (-10, 10)
            similar_players = df[
                (df['foot'] == player['foot']) &
                (df['position'] == player['position']) &
                (df['country_name'] == player['country_name']) &
                (df['Still in activity'] == 'Yes') &
                (df['height_in_cm'] >= player['height_in_cm'] + height_range[0]) &
                (df['height_in_cm'] <= player['height_in_cm'] + height_range[1])
            ]

            st.write("### Market Value Comparison with Similar Players")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=similar_players, x='player_name', y='market_value_in_millions', ax=ax, palette='pastel')
            ax.axhline(player['market_value_in_millions'], color='red', linestyle='--', linewidth=2)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
            ax.set_xlabel('Player Name', fontsize=14)
            ax.set_ylabel('Market Value in Millions (€)', fontsize=14)
            ax.set_title('Market Value Comparison with Similar Players', fontsize=16, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10, color='black')   
            st.pyplot(fig)
            st.write("The chart above compares the market value of the selected player with similar players in terms of foot, position, competition, and height (±5 cm).")

        else:
            st.write("Player not found.")
