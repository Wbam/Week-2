# utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

def load_data(file):
    """Load CSV file into a pandas DataFrame"""
    return pd.read_csv(file)

def univariate_analysis(df, column):
    """Perform univariate analysis on a specific column"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    sns.histplot(df[column], kde=True, ax=ax1)
    ax1.set_title(f'Histogram of {column}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequency')
    
    # Box plot
    sns.boxplot(x=df[column], ax=ax2)
    ax2.set_title(f'Box Plot of {column}')
    ax2.set_xlabel(column)
    
    return fig

def bivariate_analysis(df, x_column, y_column):
    """Perform bivariate analysis on two columns"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax)
    ax.set_title(f'Scatter Plot: {x_column} vs {y_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    return fig

def correlation_heatmap(df):
    """Generate a correlation heatmap for all numeric columns"""
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

def visualize_app_traffic(df):
    """Visualize top 3 most used applications and return top 10 app traffic"""
    applications = ['social_media_dl_(bytes)', 'email_dl_(bytes)', 'youtube_dl_(bytes)', 
                    'netflix_dl_(bytes)', 'gaming_dl_(bytes)']
    app_traffic = df[applications].sum()
    top_3_apps = app_traffic.nlargest(3)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_3_apps.plot(kind='bar', ax=ax, color=['blue', 'orange', 'green'])
    ax.set_title('Top 3 Most Used Applications')
    ax.set_xlabel('Application')
    ax.set_ylabel('Total Traffic (bytes)')

    return fig

def preprocess_and_aggregate(df):
    # First, combine columns
    df['TCP_retransmission'] = df[['tcp_dl_retrans._vol_(bytes)', 'tcp_ul_retrans._vol_(bytes)']].mean(axis=1)
    df['RTT'] =df[['avg_rtt_ul_(ms)', 'avg_rtt_dl_(ms)']].mean(axis=1)
    df['throughput'] = df[['avg_bearer_tp_dl_(kbps)', 'avg_bearer_tp_ul_(kbps)']].mean(axis=1)
    
    # Then, aggregate per customer
    customer_agg = df.groupby('msisdn/number').agg({
        'TCP_retransmission': 'mean',
        'RTT': 'mean',
        'handset_type': lambda x: x.mode()[0],
        'throughput': 'mean'
    }).reset_index()
    
    return customer_agg


def plot_handset_type_distributions(df):
    throughput_distribution = df.groupby('handset_type')['throughput'].mean()
    tcp_retransmission_avg =  df.groupby('handset_type')['TCP_retransmission'].mean()


    top_n = 10

    top_handsets = throughput_distribution.nlargest(top_n).index
    throughput_distribution = throughput_distribution[top_handsets]
    tcp_retransmission_avg = tcp_retransmission_avg[top_handsets]

    plt.figure(figsize=(12, 6))
    throughput_distribution.plot(kind='bar', color='skyblue')
    plt.title('Average Throughput per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average Throughput')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    tcp_retransmission_avg.plot(kind='bar', color='salmon')
    plt.title('Average TCP Retransmission per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average TCP Retransmission')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def perform_clustering(df):
    features = df[['TCP_retransmission', 'RTT', 'throughput']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    df['Cluster'] = clusters
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    print("Cluster Centers (reversed scaling):")
    print(pd.DataFrame(centers, columns=['TCP_retransmission', 'RTT', 'throughput']))

    cluster_description = df.groupby('Cluster').mean(numeric_only=True)
    print("\nCluster Descriptions:")
    print(cluster_description)

    plt.figure(figsize=(12, 8))
    plt.scatter(df['TCP_retransmission'], df['throughput'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('TCP Retransmission')
    plt.ylabel('Throughput')
    plt.title('Clustering of Users by Experience Metrics')
    plt.colorbar(label='Cluster')
    plt.show()
