import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def clean_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    print(df.head())
    print(df.columns)

    print(df.isnull().sum())
    df.dropna(inplace=True)

    print(df.dtypes)

    print(df.duplicated().sum())
    df.drop_duplicates(inplace=True)

    z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
    df = df[(abs(z_scores) < 3).all(axis=1)]

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    print(df.info())
    print(df.describe())

    df.to_csv('C:/Users/User/Desktop/10/data-2/Week2_challenge_data_source_cleaned.csv', index=False)
    
    return df

def process_engagement_data(df):
   
    df['session_duration'] = df['end_ms'] - df['start_ms']
    df['total_traffic'] = df['total_dl_(bytes)'] + df['total_ul_(bytes)']

    engagement_df = df.groupby('msisdn/number').agg(
        sessions_frequency=('start', 'count'),
        session_duration=('session_duration', 'sum'),
        total_traffic=('total_traffic', 'sum')
    ).reset_index()

    return engagement_df

def analyze_top_customers(engagement_df):
    
    top_10_sessions_frequency = engagement_df.nlargest(10, 'sessions_frequency')
    top_10_session_duration = engagement_df.nlargest(10, 'session_duration')
    top_10_total_traffic = engagement_df.nlargest(10, 'total_traffic')

    return top_10_sessions_frequency, top_10_session_duration, top_10_total_traffic

def perform_clustering(engagement_df):
    
    features = ['sessions_frequency', 'session_duration', 'total_traffic']
    scaler = StandardScaler()
    engagement_df_scaled = scaler.fit_transform(engagement_df[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    engagement_df['cluster'] = kmeans.fit_predict(engagement_df_scaled)

    cluster_stats = engagement_df.groupby('cluster').agg(
        min_sessions_frequency=('sessions_frequency', 'min'),
        max_sessions_frequency=('sessions_frequency', 'max'),
        avg_sessions_frequency=('sessions_frequency', 'mean'),
        total_sessions_frequency=('sessions_frequency', 'sum'),
        
        min_session_duration=('session_duration', 'min'),
        max_session_duration=('session_duration', 'max'),
        avg_session_duration=('session_duration', 'mean'),
        total_session_duration=('session_duration', 'sum'),
        
        min_total_traffic=('total_traffic', 'min'),
        max_total_traffic=('total_traffic', 'max'),
        avg_total_traffic=('total_traffic', 'mean'),
        total_total_traffic=('total_traffic', 'sum')
    ).reset_index()

    return engagement_df, cluster_stats

def visualize_app_traffic(df):
    applications = ['social_media_dl_(bytes)', 'email_dl_(bytes)', 'youtube_dl_(bytes)', 
                    'netflix_dl_(bytes)', 'gaming_dl_(bytes)']

    app_traffic = df.groupby('msisdn/number')[applications].sum().reset_index()

    top_10_app_traffic = {}
    for app in applications:
        top_10_app_traffic[app] = app_traffic.nlargest(10, app)

    total_app_traffic = app_traffic[applications].sum()
    top_3_apps = total_app_traffic.nlargest(3)

    top_3_apps.plot(kind='bar', color=['blue', 'orange', 'green'])
    plt.title('Top 3 Most Used Applications')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (bytes)')
    plt.show()

    return top_10_app_traffic

def plot_elbow_method(engagement_df_scaled):
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(engagement_df_scaled)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.show()
