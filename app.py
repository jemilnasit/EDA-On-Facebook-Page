import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

st.title("CSV File Upload and Analyse Data")
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

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        
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
            
            km = KMeans(n_clusters=k_point, init='k-means++', n_init=10, random_state=0)
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
            
            ss_km = silhouette_score(X, df['cluster'])
            st.write(f'🌟**Silhouette Score: {ss_km}**')
            
            df.drop(['cluster'], axis=1, inplace=True)
            
        with st.container():
            st.subheader("Hierarchical Clustering Model")
            
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(df)
            
            pca = PCA(n_components=2)
            pca_scaled = pca.fit_transform(x_scaled)
            
            st.write('click on the button below for hierarchical clustering Dendrogram:')
            if st.button('****📊****'):
                fig, ax = plt.subplots()
                dendrogram = sc.dendrogram(sc.linkage(pca_scaled, method='ward'), ax=ax)
                ax.set_title('Hierarchical Clustering Dendrogram')
                ax.set_xlabel('Samples')
                ax.set_ylabel('Distance')
                st.pyplot(fig)
            
            st.write('show the graph and Look for the largest vertical gap (linkage distance) between merged clusters. Draw a horizontal line that cuts the dendrogram through this largest vertical gap without intersecting any merges.Count the number of vertical lines the horizontal line intersects — this is your k.')
            k_point_hc = st.number_input('🔢Enter the number of clusters or k value:', min_value=1, max_value=30, value=3)
            hc = AgglomerativeClustering(n_clusters=k_point_hc, linkage='ward')
            hc.fit(x_scaled)
            
            fig, ax = plt.subplots()
            ax.scatter(pca_scaled[:, 0], pca_scaled[:, 1], c=hc.labels_, cmap='viridis', s=50)
            ax.set_title('Hierarchical Clustering')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            st.pyplot(fig)
            
            ss_hc = silhouette_score(x_scaled, hc.labels_)
            st.write(f'🌟**Silhouette Score: {ss_hc}**')
        
        with st.container():
            st.subheader("DBSCAN Clustering Model")
            
            st.write('eps - Maximum distance between two points for one to be considered as in the neighborhood of the other')
            eps_value = st.number_input('🔢Enter the eps value:', min_value=0.1, max_value=10.0, value=1.3, step=0.1)
            
            st.write('min_samples - The number of samples in a neighborhood for a point to be considered as a core point') 
            min_samples_value = st.number_input('🔢Enter the min_samples value:', min_value=1, max_value=100, value=5, step=1)   
            
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            dbscan.fit(x_scaled)
            labels = dbscan.labels_
            
            d = labels != -1
            
            pca = PCA(n_components=2)
            pca_scaled = pca.fit_transform(x_scaled)
            pca_filtered = pca_scaled[d]
            labels_filtered = labels[d]
            
            fig, ax = plt.subplots()
            ax.scatter(pca_filtered[:, 0], pca_filtered[:, 1], c=labels_filtered, cmap='viridis', s=50)
            ax.set_title('DBSCAN Clustering')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            st.pyplot(fig)
            
            ss_dbscan = silhouette_score(x_scaled[d], labels_filtered)
            st.write(f'🌟**Silhouette Score: {ss_dbscan}**')
            
        st.write('*****select the best model based on the silhouette score*****')
        st.write('click on the button below for silhouette score comparison:')
        if st.button('****📊****',key='k2'):    
            score = [ss_km, ss_hc, ss_dbscan]
            method = ['KMeans', 'Agglomerative', 'DBSCAN']
            fig, ax = plt.subplots()
            ax.bar(method, score, color=['green', 'gray', 'red'], width=0.4)
            ax.set_title('Silhouette Score Comparison')
            ax.set_ylabel('Silhouette Score')
            ax.set_xlabel('Clustering Method')
            st.pyplot(fig)
            index = score.index(max(score))
            st.write('*****The higher the silhouette score, the better the clustering.*****')
            st.write(f'♦️***best model is: {method[index]}***')
            
    except Exception as e:
        st.error(f"Error is: {e}")
        
