# main.py

import streamlit as st
import pandas as pd
from utils import load_data, univariate_analysis, bivariate_analysis, correlation_heatmap,visualize_app_traffic,plot_handset_type_distributions,preprocess_and_aggregate,perform_clustering

def main():
    st.title('CSV Data Analysis Dashboard')

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success('File successfully uploaded and loaded!')

        # Display raw data
        st.subheader('Raw Data')
        st.write(df.head())

        # Univariate Analysis
        st.subheader('Univariate Analysis')
        column = st.selectbox('Select a column for univariate analysis:', df.columns)
        fig = univariate_analysis(df, column)
        st.pyplot(fig)

        # Summary statistics
        st.subheader('Summary Statistics')
        st.write(df[column].describe())

        # Bivariate Analysis
        st.subheader('Bivariate Analysis')
        x_column = st.selectbox('Select X-axis column:', df.columns)
        y_column = st.selectbox('Select Y-axis column:', df.columns)
        fig = bivariate_analysis(df, x_column, y_column)
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader('Correlation Heatmap')
        fig = correlation_heatmap(df)
        st.pyplot(fig)
        
         # App Traffic Visualization
        st.subheader('App Traffic Visualization')
        fig = visualize_app_traffic(df)
        st.pyplot(fig)

        
        # Plot handset type distributions
        st.subheader('Plot handset type distributions')
        data = preprocess_and_aggregate(df)
        fig = plot_handset_type_distributions(data)
        st.pyplot(fig)

        # Perform clustering
        st.subheader('Plot handset type distributions')
        data = preprocess_and_aggregate(df)
        fig = perform_clustering(data)
        st.pyplot(fig)
        
if __name__ == '__main__':
    main()