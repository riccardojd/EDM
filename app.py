import streamlit as st
import numpy as np
import pickle

# Load pre-trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Function to classify EDM subgenres
def classify_genre(audio_features):
    features = np.array([[
        audio_features['acousticness'],
        audio_features['danceability'],
        audio_features['duration_ms'],
        audio_features['energy'],
        audio_features['instrumentalness'],
        audio_features['liveness'],
        audio_features['loudness'],
        audio_features['mode'],
        audio_features['speechiness'],
        audio_features['tempo'],
        audio_features['valence']
    ]])
    features_scaled = scaler.transform(features)
    genre = model.predict(features_scaled)
    return genre[0]

# Streamlit app
st.title('EDM Genre Classification')

# Input fields for song features
acousticness = st.number_input('Acousticness', min_value=0.0, max_value=1.0, step=0.01)
danceability = st.number_input('Danceability', min_value=0.0, max_value=1.0, step=0.01)
duration_ms = st.number_input('Duration (ms)', min_value=0)
energy = st.number_input('Energy', min_value=0.0, max_value=1.0, step=0.01)
instrumentalness = st.number_input('Instrumentalness', min_value=0.0, max_value=1.0, step=0.01)
liveness = st.number_input('Liveness', min_value=0.0, max_value=1.0, step=0.01)
loudness = st.number_input('Loudness (dB)', min_value=-60.0, max_value=0.0, step=0.1)
mode = st.selectbox('Mode', options=[0, 1])
speechiness = st.number_input('Speechiness', min_value=0.0, max_value=1.0, step=0.01)
tempo = st.number_input('Tempo (BPM)', min_value=0.0, max_value=300.0, step=0.1)
valence = st.number_input('Valence', min_value=0.0, max_value=1.0, step=0.01)

audio_features = {
    'acousticness': acousticness,
    'danceability': danceability,
    'duration_ms': duration_ms,
    'energy': energy,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'loudness': loudness,
    'mode': mode,
    'speechiness': speechiness,
    'tempo': tempo,
    'valence': valence
}

if st.button('Classify Genre'):
    genre = classify_genre(audio_features)
    st.write(f'The genre of the track is: {genre}')
