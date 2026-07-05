import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# 1. PREPROCESSING MODULE

def preprocess_dataframe(df):
    """Processes the dataframe in-memory and ensures absolute elimination of NaNs."""
    cols = ["id", "name", "city", "rating", "rating_count", "cost", "cuisine", "lic_no", "link", "address", "menu"]
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].drop_duplicates().copy()
    
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

    df["cost"] = (
    df["cost"]
    .astype(str)
    .str.replace(r"[^\d]", "", regex=True)
)

    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    
    df.dropna(subset=["name", "city", "cuisine"], inplace=True)
    
    rating_fallback = df["rating"].median()
    df["rating"] = df["rating"].fillna(rating_fallback if pd.notna(rating_fallback) else 0.0)
    
    count_fallback = df["rating_count"].median()
    df["rating_count"] = df["rating_count"].fillna(count_fallback if pd.notna(count_fallback) else 0)
    
    cost_fallback = df["cost"].median()
    df["cost"] = df["cost"].fillna(cost_fallback if pd.notna(cost_fallback) else 0.0)
    
    df["cuisine"] = df["cuisine"].astype(str).apply(lambda x: ", ".join(x.split(",")[:2]))
    df.reset_index(drop=True, inplace=True)
    
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X = enc.fit_transform(df[["city", "cuisine"]])
    encoded = pd.DataFrame(X, columns=enc.get_feature_names_out(["city", "cuisine"]))
    
    encoded["rating"] = df["rating"].values
    encoded["rating_count"] = df["rating_count"].values
    encoded["cost"] = df["cost"].values
    
    encoded.fillna(0.0, inplace=True)
    
    return df, encoded

# 2. RECOMMENDATION ENGINE MODULE

class RestaurantRecommender:

    def __init__(self, cleaned_df, encoded_df):

        self.cleaned_df = cleaned_df
        self.encoded_df = encoded_df

    def search_restaurant(self, keyword):
        keyword = str(keyword).lower()
        return self.cleaned_df[
            self.cleaned_df["name"].str.lower().str.contains(keyword, na=False)
        ]

    def filter_restaurants(self, city=None, cuisine=None, min_rating=0, max_cost=None):
        df = self.cleaned_df.copy()

        if city:
            df = df[df["city"] == city]

        if cuisine:
            df = df[df["cuisine"].str.contains(cuisine, case=False, na=False)]

        df = df[df["rating"] >= min_rating]

        if max_cost is not None:
            df = df[df["cost"] <= max_cost]

        return df.reset_index(drop=True)

    def recommend_by_name(self, restaurant_name, top_n=5):
        matches = self.cleaned_df[
            self.cleaned_df["name"].str.lower() == restaurant_name.lower()
        ]

        if matches.empty:
            raise ValueError("Restaurant not found.")

        idx = matches.index[0]

        query = self.encoded_df.iloc[[idx]]

        scores = cosine_similarity(
    query,
    self.encoded_df
).flatten()

        scores = list(enumerate(scores))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for index, score in scores:
            if index == idx:
                continue 
            recommendations.append(self.cleaned_df.iloc[index])
            if len(recommendations) == top_n:
                break

        return pd.DataFrame(recommendations)

# 3. INTERFACE UTILITIES MODULE

def show_metrics(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Restaurants", len(df))
    c2.metric("Cities", df["city"].nunique())
    c3.metric("Cuisines", df["cuisine"].nunique())
    c4.metric("Average Rating", round(df["rating"].mean(), 2))

def recommendation_card(row):
    with st.container():
        st.markdown(f"### 🍽️ {row['name']}")
        c1, c2 = st.columns(2)

        c1.write(f"**City:** {row['city']}")
        c1.write(f"**Cuisine:** {row['cuisine']}")
        c1.write(f"**Rating:** ⭐ {row['rating']}")

        c2.write(f"**Cost:** ₹ {row['cost']}")
        c2.write(f"**Address:** {row['address'] if pd.notna(row['address']) else 'N/A'}")

        if "link" in row and pd.notna(row['link']):
            st.markdown(f"[Open Swiggy Page]({row['link']})")
        st.markdown("---")

def show_recommendations(df):
    if df.empty:
        st.warning("No recommendations found.")
        return
    for _, row in df.iterrows():
        recommendation_card(row)

def inject_css():
    st.markdown(
        """
        <style>
        div[data-testid="metric-container"]{
            background:#f4f4f4;
            border-radius:10px;
            padding:15px;
        }
        .stButton>button{
            width:100%;
            border-radius:10px;
            height:45px;
            font-size:16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 4. MAIN STREAMLIT APPLICATION ENTRYPOINT

st.set_page_config(page_title="Swiggy Recommendation", layout="wide")
inject_css()


st.sidebar.title("🍽️ Swiggy Recommendation")
page = st.sidebar.radio("Navigation", ["Home", "Recommend", "Insights", "About"])

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload swiggy.csv", type=["csv"])

if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.cleaned_df = None
    st.session_state.encoded_df = None

if uploaded and not st.session_state.data_processed:
    with st.spinner("Processing large dataset... please wait..."):
        raw_df = pd.read_csv(uploaded)
        clean_df, enc_df = preprocess_dataframe(raw_df)
        
        st.session_state.cleaned_df = clean_df
        st.session_state.encoded_df = enc_df
        st.session_state.data_processed = True
        st.rerun()

if not uploaded and st.session_state.data_processed:
    st.session_state.data_processed = False
    st.session_state.cleaned_df = None
    st.session_state.encoded_df = None

has_data = st.session_state.data_processed
if has_data:
    recommender = RestaurantRecommender(st.session_state.cleaned_df, st.session_state.encoded_df)
    df = recommender.cleaned_df


if page == "Home":
    if not has_data:
        st.info("⚠️ Please upload a 'swiggy.csv' dataset file via the sidebar to view the dashboard.")
    else:
        st.title("🍕 Swiggy Restaurant Recommendation Dashboard")
        show_metrics(df)
        
        st.subheader("Explore Dataset View")
        search_kw = st.text_input("Quick Search by Restaurant Name", "")
        if search_kw:
            display_df = recommender.search_restaurant(search_kw)
        else:
            display_df = df.head(20)
            
        st.dataframe(display_df, use_container_width=True)

elif page == "Recommend":
    if not has_data:
        st.info("⚠️ Please upload a 'swiggy.csv' dataset file via the sidebar to get recommendations.")
    else:
        st.title("Restaurant Recommendation Engine")
        
        city = st.selectbox("Select City", sorted(df["city"].unique()))
        available_cuisines = sorted(df[df["city"] == city]["cuisine"].unique())
        cuisine = st.selectbox("Select Cuisine", available_cuisines)
        
        rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
        cost = st.slider("Maximum Cost", int(df["cost"].min()), int(df["cost"].max()), int(df["cost"].max()))
        
        filtered = recommender.filter_restaurants(city, cuisine, rating, cost)
        
        if filtered.empty:
            st.warning("No restaurants match your filters. Try relaxing your constraints!")
        else:
            name = st.selectbox("Pick a Restaurant to find matches for", filtered["name"].unique())
            top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)
            
            if st.button("Generate Recommendations"):
                try:
                    results = recommender.recommend_by_name(name, top_n=top_n)
                    st.subheader(f"Top Pick Matches for '{name}':")
                    show_recommendations(results)
                except Exception as e:
                    st.error(f"Error producing recommendations: {e}")

elif page == "Insights":
    if not has_data:
        st.info("⚠️ Please upload a 'swiggy.csv' dataset file via the sidebar to view insights charts.")
    else:
        st.title("📊 Data Insights")
        st.subheader("Top Rated Cities")
        city_ratings = df.groupby("city")["rating"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(city_ratings)
        
        st.subheader("Cost Distribution Profile")
        st.area_chart(df["cost"].value_counts().sort_index().head(50))

elif page == "About":
    st.title("ℹ️ About This Application")
    st.markdown("""
    This application leverages **Content-Based Filtering via Cosine Similarity** to recommend similar dining options based on location profiles, cost indices, and cuisine descriptions pulled directly from Swiggy data.
    
    ### How to use:
    1. Drop your raw `swiggy.csv` file into the file uploader tool on the left sidebar.
    2. Head over to the **Home** tab to browse or search through entries.
    3. Use the **Recommend** tab to isolate choices by city or budget metrics and discover alternatives.
    """)
