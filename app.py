import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page configuration
st.set_page_config(page_title="Clustering Analysis App", layout="wide")

# Apply custom style
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2e4053;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stNumberInput>div>div>input {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Machine Learning Clustering Analyzer")
st.write("""
Upload a **CSV file** and analyze it with **KMeans**, **Hierarchical**, and **DBSCAN** clustering algorithms. 
Get recommendations based on **Silhouette Score** and explore detailed visualizations.
""")

uploaded_file = st.file_uploader("📁 Upload a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        with st.expander("🔍 Dataset Overview", expanded=True):
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", df.columns.tolist())
            st.dataframe(df)

        with st.expander("🧹 Data Cleaning and Encoding"):
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())

            st.write("**Basic Statistics:**")
            st.dataframe(df.describe())

            le = LabelEncoder()
            for column in df.columns:
                if df[column].dtype == 'object':
                    df[column] = le.fit_transform(df[column])

            st.success("✅ Data Cleaned & Encoded")
            st.dataframe(df)

        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        pca = PCA(n_components=2)
        pca_scaled = pca.fit_transform(scaled_df)

        with st.container():
            st.header("⚙️ KMeans Clustering")
            wcss = []
            for i in range(1, 31):
                kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
                kmeans.fit(df)
                wcss.append(kmeans.inertia_)

            kl = KneeLocator(range(1, 31), wcss, curve='convex', direction='decreasing')
            optimal_k = kl.elbow or 3
            st.success(f"📍 Optimal k found: {optimal_k}")

            if st.button("📉 Show Elbow Plot"):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, 31), wcss)
                ax.axvline(x=optimal_k, color='r', linestyle='--')
                ax.set_title('Elbow Method')
                ax.set_xlabel('k')
                ax.set_ylabel('WCSS')
                st.pyplot(fig)

            km_model = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=0)
            df['kmeans_cluster'] = km_model.fit_predict(df)

            with st.expander("📊 KMeans Cluster Summary"):
                st.write(df['kmeans_cluster'].value_counts())
                st.dataframe(df.groupby('kmeans_cluster').mean())

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(pca_scaled[:, 0], pca_scaled[:, 1], c=df['kmeans_cluster'], cmap='viridis', s=60)
            ax.set_title("KMeans Clustering")
            st.pyplot(fig)

            kmeans_score = silhouette_score(df.drop('kmeans_cluster', axis=1), df['kmeans_cluster'])
            st.success(f"KMeans Silhouette Score: {kmeans_score:.4f}")

        with st.container():
            st.header("🔗 Hierarchical Clustering")
            if st.button("📈 Show Dendrogram"):
                fig, ax = plt.subplots(figsize=(8, 4))
                sch.dendrogram(sch.linkage(pca_scaled, method='ward'), ax=ax)
                ax.set_title('Dendrogram')
                st.pyplot(fig)

            hc_k = st.slider("Choose number of clusters", 2, 10, 3)
            hc_model = AgglomerativeClustering(n_clusters=hc_k)
            df['hc_cluster'] = hc_model.fit_predict(scaled_df)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(pca_scaled[:, 0], pca_scaled[:, 1], c=df['hc_cluster'], cmap='plasma', s=60)
            ax.set_title("Hierarchical Clustering")
            st.pyplot(fig)

            hc_score = silhouette_score(scaled_df, df['hc_cluster'])
            st.success(f"Hierarchical Silhouette Score: {hc_score:.4f}")

        with st.container():
            st.header("📍 DBSCAN Clustering")
            eps = st.slider("Select eps", 0.1, 5.0, 1.0, 0.1)
            min_samples = st.slider("Select min_samples", 1, 20, 5)
            db_model = DBSCAN(eps=eps, min_samples=min_samples)
            df['dbscan_cluster'] = db_model.fit_predict(scaled_df)

            valid = df['dbscan_cluster'] != -1
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(pca_scaled[valid, 0], pca_scaled[valid, 1], c=df['dbscan_cluster'][valid], cmap='Accent', s=60)
            ax.set_title("DBSCAN Clustering")
            st.pyplot(fig)

            db_score = silhouette_score(scaled_df[valid], df['dbscan_cluster'][valid]) if valid.any() else 0
            st.success(f"DBSCAN Silhouette Score: {db_score:.4f}")

        with st.container():
            st.header("🏆 Model Comparison")
            scores = [kmeans_score, hc_score, db_score]
            labels = ['KMeans', 'Hierarchical', 'DBSCAN']

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=labels, y=scores, palette='pastel', ax=ax)
            ax.set_title("Silhouette Score Comparison")
            ax.set_ylabel("Silhouette Score")
            st.pyplot(fig)

            best_idx = np.argmax(scores)
            st.success(f"🎉 Best clustering model: **{labels[best_idx]}** with score **{scores[best_idx]:.4f}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
