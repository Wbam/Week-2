import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_duplicated_bearer_ids(df):
    duplicate_bearer_ids = df['Bearer Id'].value_counts()[df['Bearer Id'].value_counts() > 1]

    duplicates_sample = df[df['Bearer Id'].isin(duplicate_bearer_ids.index)].head(10)
    print("Sample of records with duplicated 'Bearer Id':\n", duplicates_sample)

    duration_analysis = df[df['Bearer Id'].isin(duplicate_bearer_ids.index)]
    duration_stats = duration_analysis.groupby('Bearer Id')['Dur. (ms)'].describe()
    print("\nDuration analysis for each duplicated 'Bearer Id':\n", duration_stats)

    # Aggregate data for each duplicated 'Bearer Id'
    aggregated_data = duration_analysis.groupby('Bearer Id').agg({
        'Dur. (ms)': 'sum',
        'Total UL (Bytes)': 'sum',
        'Total DL (Bytes)': 'sum',
    }).reset_index()
    print("\nAggregated data for each duplicated 'Bearer Id':\n", aggregated_data.head())

    return duplicates_sample, duration_stats, aggregated_data

def additional_aggregations(df):
    # Number of xDR sessions per user
    xdr_sessions_per_user = df.groupby('IMSI')['Bearer Id'].count().reset_index()
    xdr_sessions_per_user.columns = ['IMSI', 'Number of xDR Sessions']
    print("Number of xDR sessions per user:\n", xdr_sessions_per_user.head())

    # Aggregated session duration per user
    session_duration = df.groupby('Bearer Id')['Dur. (ms)'].sum().reset_index()
    session_duration.columns = ['Bearer Id', 'Total Session Duration (ms)']
    print("Aggregated session duration per user:\n", session_duration.head())

    # Aggregated total download and upload data per user
    data_usage = df.groupby('Bearer Id').agg({
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()
    data_usage.columns = ['Bearer Id', 'Total Download Data (Bytes)', 'Total Upload Data (Bytes)']
    print("Aggregated total download and upload data per user:\n", data_usage.head())

    # Add total data volume column
    df['Total Data Volume (Bytes)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

    # Aggregate total data volume per user
    total_data_volume = df.groupby('Bearer Id').agg({
        'Total Data Volume (Bytes)': 'sum'
    }).reset_index()
    print("Aggregated total data volume per user:\n", total_data_volume.head())
    
    return xdr_sessions_per_user, session_duration, data_usage, total_data_volume
# Task 1.2
def generate_data():
    df = pd.DataFrame({
        'duration_ms': np.random.normal(loc=77536.33, scale= 527374.80, size=100),
        'social_media': np.random.normal(loc=914034.26, scale= 527374.79, size=100),
        'google': np.random.normal(loc=3903418.18, scale=2249303.92, size=100),
        'email': np.random.normal(loc=1128210.90, scale=657486.60, size=100),
        'youtube': np.random.normal(loc=11385539.02, scale=6539686.095, size=100),
        'netflix': np.random.normal(loc=11347999.45, scale=6496748.86, size=100),
        'gaming': np.random.normal(loc= 215536254.41, scale=125160265.12, size=100),
        'total_data': np.random.normal(loc=248334438.39, scale= 128465540.17, size=100),
        'other_data': np.random.normal(loc=216507959.65, scale=122720803.74, size=100),})
    return df
def analyze_data(df):
    # 1. Describe all relevant variables and associated data types
    print("Data Types:\n", df.dtypes)

    # 2. Handle missing values
    df.fillna(df.mean(), inplace=True)

    # 3. Identify and treat outliers
    for column in df.select_dtypes(include=[np.number]).columns:
        mean = df[column].mean()
        std_dev = df[column].std()
        outliers = (df[column] > mean + 3 * std_dev) | (df[column] < mean - 3 * std_dev)
        df.loc[outliers, column] = mean

    # 4. Variable Transformations: Segment users into top five decile classes based on total duration
    df['decile'] = pd.qcut(df['duration_ms'], 10, labels=False) + 1
    decile_data = df.groupby('decile')['total_data'].sum().reset_index()
    print("\nTotal Data per Decile Class:\n", decile_data)

    # 5. Basic Metrics Analysis
    basic_metrics = df.describe()
    print("\nBasic Metrics:\n", basic_metrics)

    # 6. Non-Graphical Univariate Analysis
    univariate_analysis = df.describe(include=[np.number])
    print("\nUnivariate Analysis:\n", univariate_analysis)

    # 7. Graphical Univariate Analysis
    num_columns = len(df.select_dtypes(include=[np.number]).columns)
    cols = 3  # Number of columns in subplot grid
    rows = (num_columns + cols - 1) // cols  # Calculate number of rows needed

    plt.figure(figsize=(12, 4 * rows))
    for i, column in enumerate(df.select_dtypes(include=[np.number]).columns):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
    plt.tight_layout()
    plt.show()

    # 8. Bivariate Analysis
    plt.figure(figsize=(12, 6))
    sns.pairplot(df[['social_media', 'google', 'email', 'youtube', 'netflix', 'gaming', 'total_data', 'other_data']])
    plt.suptitle('Pairplot of Application Data vs Total Data', y=1.02)
    plt.show()

    # 9. Correlation Analysis
    correlation_matrix = df[['social_media', 'google', 'email', 'youtube', 'netflix', 'gaming', 'other_data', 'total_data']].corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)

    # 10. Dimensionality Reduction: Principal Component Analysis
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[['social_media', 'google', 'email', 'youtube', 'netflix', 'gaming', 'other_data', 'total_data']])
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    print("\nPrincipal Component Analysis Results:")
    print(f"Explained Variance Ratio: {explained_variance}")

    # Interpretation
    print("\nPCA Interpretation:")
    print("1. PC1 explains {:.2f}% of the variance.".format(explained_variance[0] * 100))
    print("2. PC2 explains {:.2f}% of the variance.".format(explained_variance[1] * 100))
    print("3. The combination of PC1 and PC2 explains {:.2f}% of the total variance.".format(sum(explained_variance) * 100))
    print("4. PCA can help in reducing dimensionality while retaining most of the variance in the data.")
