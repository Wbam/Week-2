import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(data):
    data['TCP DL Retrans. Vol (Bytes)'] = data['TCP DL Retrans. Vol (Bytes)'].fillna(data['TCP DL Retrans. Vol (Bytes)'].mean())
    data['TCP UL Retrans. Vol (Bytes)'] = data['TCP UL Retrans. Vol (Bytes)'].fillna(data['TCP UL Retrans. Vol (Bytes)'].mean())
    data['Avg RTT DL (ms)'] = data['Avg RTT DL (ms)'].fillna(data['Avg RTT DL (ms)'].mean())
    data['Avg RTT UL (ms)'] = data['Avg RTT UL (ms)'].fillna(data['Avg RTT UL (ms)'].mean())
    data['Avg Bearer TP DL (kbps)'] = data['Avg Bearer TP DL (kbps)'].fillna(data['Avg Bearer TP DL (kbps)'].mean())
    data['Avg Bearer TP UL (kbps)'] = data['Avg Bearer TP UL (kbps)'].fillna(data['Avg Bearer TP UL (kbps)'].mean())
    data['Handset Type'] = data['Handset Type'].fillna(data['Handset Type'].mode()[0])
    return data

def combine_columns(data):
    data['TCP_retransmission'] = data[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean(axis=1)
    data['RTT'] = data[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean(axis=1)
    data['throughput'] = data[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean(axis=1)
    return data

def remove_outliers(data):
    for col in ['TCP_retransmission', 'RTT', 'throughput']:
        mean = data[col].mean()
        std_dev = data[col].std()
        data[col] = np.where(
            (data[col] < mean - 3 * std_dev) | (data[col] > mean + 3 * std_dev),
            mean,
            data[col]
        )
    return data

def aggregate_per_customer(data):
    customer_agg = data.groupby('MSISDN/Number').agg({
        'TCP_retransmission': 'mean',
        'RTT': 'mean',
        'Handset Type': lambda x: x.mode()[0],
        'throughput': 'mean'
    }).reset_index()
    customer_agg.to_csv('customer_aggregated_data.csv', index=False)
    return customer_agg

def compute_top_bottom_frequent(column):
    top_10 = column.sort_values(ascending=False).head(10)
    bottom_10 = column.sort_values().head(10)
    most_frequent = column.mode().head(10)
    return top_10, bottom_10, most_frequent

def print_frequent_values(data):
    tcp_top_10, tcp_bottom_10, tcp_most_frequent = compute_top_bottom_frequent(data['TCP_retransmission'])
    rtt_top_10, rtt_bottom_10, rtt_most_frequent = compute_top_bottom_frequent(data['RTT'])
    throughput_top_10, throughput_bottom_10, throughput_most_frequent = compute_top_bottom_frequent(data['throughput'])

    print("\nTop 10 TCP retransmission values:")
    print(tcp_top_10)
    print("\nBottom 10 TCP retransmission values:")
    print(tcp_bottom_10)
    print("\nMost Frequent TCP retransmission values:")
    print(tcp_most_frequent)

    print("\nTop 10 RTT values:")
    print(rtt_top_10)
    print("\nBottom 10 RTT values:")
    print(rtt_bottom_10)
    print("\nMost Frequent RTT values:")
    print(rtt_most_frequent)

    print("\nTop 10 Throughput values:")
    print(throughput_top_10)
    print("\nBottom 10 Throughput values:")
    print(throughput_bottom_10)
    print("\nMost Frequent Throughput values:")
    print(throughput_most_frequent)

def plot_handset_type_distributions(customer_agg):
    throughput_distribution = customer_agg.groupby('Handset Type')['throughput'].mean()
    tcp_retransmission_avg = customer_agg.groupby('Handset Type')['TCP_retransmission'].mean()

    print("Distribution of average throughput per handset type:")
    print(throughput_distribution)

    print("Average TCP retransmission per handset type:")
    print(tcp_retransmission_avg)

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

def perform_clustering(customer_agg):
    features = customer_agg[['TCP_retransmission', 'RTT', 'throughput']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    customer_agg['Cluster'] = clusters
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    print("Cluster Centers (reversed scaling):")
    print(pd.DataFrame(centers, columns=['TCP_retransmission', 'RTT', 'throughput']))

    cluster_description = customer_agg.groupby('Cluster').mean(numeric_only=True)
    print("\nCluster Descriptions:")
    print(cluster_description)

    plt.figure(figsize=(12, 8))
    plt.scatter(customer_agg['TCP_retransmission'], customer_agg['throughput'], c=customer_agg['Cluster'], cmap='viridis')
    plt.xlabel('TCP Retransmission')
    plt.ylabel('Throughput')
    plt.title('Clustering of Users by Experience Metrics')
    plt.colorbar(label='Cluster')
    plt.show()

def main():
    file_path = 'C:/Users/User/Desktop/10/data-2/Week2_challenge_data_source(CSV).csv'
    data = load_data(file_path)
    data = handle_missing_values(data)
    data = combine_columns(data)
    data = remove_outliers(data)
    customer_agg = aggregate_per_customer(data)

    print("Aggregation complete. Data saved to 'customer_aggregated_data.csv'.")

    print_frequent_values(data)
    plot_handset_type_distributions(customer_agg)
    perform_clustering(customer_agg)

if __name__ == "__main__":
    main()
