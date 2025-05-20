import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
        
        
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        
