import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- STEP 1: DATA CLEANING ---
def load_and_clean_data(file_path):
    df = pd.read_csv("C:/Users/ashwi/Downloads/sql session 1/swiggy.csv")

    cols = ['name', 'city', 'rating', 'cost', 'cuisine']
    df = df[cols].dropna().drop_duplicates().reset_index(drop=True)
   
    df['cuisine'] = df['cuisine'].apply(lambda x: ", ".join(x.split(',')[:2]))
    
    df.to_csv("cleaned_data.csv", index=False)
    return df

#MEMORY-EFFICIENT PREPROCESSING
def preprocess_data(df):
  
    encoder = OneHotEncoder(sparse_output=True) 
    
    encoded_sparse = encoder.fit_transform(df[['city', 'cuisine']])
    
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
        
    return encoded_sparse, encoder

# RECOMMENDATION 
def get_recommendations(target_index, sparse_matrix, original_df, top_n=5):

    sim_scores = cosine_similarity(sparse_matrix[target_index], sparse_matrix).flatten()
    
    similar_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
    
    return original_df.iloc[similar_indices]

# --- STREAMLIT UI ---
st.title("🍕🍔 Swiggy Restaurant Recommendation System 🍜🥗")

uploaded_file = st.sidebar.file_uploader("C:/Users/ashwi/Downloads/sql session 1/swiggy.csv", type="csv")

if uploaded_file is not None:

    df = load_and_clean_data(uploaded_file)
    sparse_mat, encoder = preprocess_data(df)

    city = st.selectbox("Select City", df['city'].unique())
    cuisine = st.selectbox("Select Cuisine", df['cuisine'].unique())

    selected_res = st.selectbox("Pick a restaurant:", df['name'].unique())

    if st.button("Recommend"):
        idx = df[df['name'] == selected_res].index[0]
        results = get_recommendations(idx, sparse_mat, df)
        st.dataframe(results)

else:
    st.info("C:/Users/ashwi/Downloads/sql session 1/swiggy.csv")