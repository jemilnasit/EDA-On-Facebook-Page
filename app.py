import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.title("CSV File Upload and Display")
st.write("Upload a CSV file to display its contents.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try: 
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the dataframe
        st.write("DataFrame:")
        st.dataframe(df)

        # Display the shape of the dataframe
        st.write("Shape of the DataFrame:")
        st.write(df.shape)

        # Display the columns of the dataframe
        st.write("Columns in the DataFrame:")
        st.write(df.columns.tolist())

        df.dropna(inplace=True)
        # Display basic statistics of the dataframe
        st.write("Basic Statistics of the DataFrame:")
        st.dataframe(df.describe())
                
        le_df = LabelEncoder()
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = le_df.fit_transform(df[column])
                
        st.write("Label Encoded DataFrame:")
        st.dataframe(df)
        
        with st.container():
            st.subheader("KMeans Clustering Model")
            
            wcss = []

            st.write('*K point for K-Means Clustering using Elbow Method:*')
            for i in range(1, 31):
                km = KMeans(n_clusters=i, init='k-means++',n_init=10, random_state=0)
                model = km.fit(df)
                wcss.append(km.inertia_)

            kl = KneeLocator(range(1, 31), wcss, curve='convex', direction='decreasing')
            k_point = kl.elbow
            st.write(f'➡️ k point is : {k_point}')
        
            st.write('♦️ click on the button below for elbow plot:')
            if st.button('Elbow plot'):
                fig, ax = plt.subplots()
                ax.plot(range(1, 31), wcss)
                ax.axvline(x=k_point, color='r', linestyle='--')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('WCSS')
                ax.set_title('Elbow Method for Optimal k')
                st.pyplot(fig)
            
            km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=0)
            model = km.fit(df)
            df['cluster'] = model.labels_
            
            with st.expander("Show cluster summary"):
                
                st.write('values in each cluster:')
                x = df['cluster'].value_counts()
                x
                
                st.write("Cluster Summary using mean:")
                cluster_summary = df.groupby('cluster')[df.columns].mean()
                cluster_summary.T
                
                st.write("Cluster Summary using mode:")
                cluster_summary_mode = df.groupby('cluster')[df.columns].agg(lambda x: pd.Series.mode(x).iloc[0])
                cluster_summary_mode.T

            st.write('***kmeans clustering plot***')
            
            pca = PCA(n_components=2)
            X = df.drop(['cluster'], axis=1)
            pca_scaled = pca.fit_transform(X)
            
            fig, ax = plt.subplots()
            ax.scatter(pca_scaled[:, 0], pca_scaled[:, 1], c=df['cluster'], cmap='viridis', s=50)
            ax.set_title('KMeans Clustering')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            st.pyplot(fig)
            
            ss_km = silhouette_score(df, df['cluster'])
            st.write(f'🌟**Silhouette Score: {ss_km}**')
            
    except Exception as e:
        st.error(f"Error is: {e}")
        
