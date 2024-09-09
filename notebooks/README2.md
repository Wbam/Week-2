# Task 3: Experience Analytics

## Overview

This task involves analyzing user experience metrics in the telecommunication industry. We focus on key network performance indicators such as TCP retransmissions, Round Trip Time (RTT), and throughput, as well as handset types to understand and improve user experiences.

The analysis includes:

1. Aggregating user experience metrics.
2. Identifying top, bottom, and most frequent values for TCP retransmission, RTT, and throughput.
3. Analyzing and visualizing the distribution of average throughput and TCP retransmission per handset type.
4. Performing k-means clustering to segment users based on their experience metrics.

## Instructions

### Prerequisites

- Python 3.x
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

### Data

Ensure the dataset is available in CSV format and loaded correctly in the script. Update the file path as needed in the script.

### Running the Code

1. **Load and Preprocess Data:**
   Ensure that the dataset is loaded and missing values are handled as described in Task 3.1.

2. **Execute Aggregation and Analysis:**

   - **Task 3.1**: Aggregate the data to compute average TCP retransmission, RTT, throughput, and handset type per customer.
   - **Task 3.2**: Compute and list the top 10, bottom 10, and most frequent values for TCP retransmission, RTT, and throughput.
   - **Task 3.3**: Analyze and visualize the distribution of average throughput and TCP retransmission per handset type.

3. **Perform Clustering:**
   - **Task 3.4**: Use k-means clustering (with k=3) to segment users based on experience metrics and provide cluster descriptions.

### Code Execution

Run the provided Python script to perform the following steps:

- Load the dataset and preprocess it.
- Aggregate metrics and handle missing values.
- Compute top, bottom, and frequent values for key metrics.
- Generate and display plots for throughput and TCP retransmission distributions.
- Perform k-means clustering and visualize the clustering results.

### Results

- **Aggregation**: Averages of TCP retransmission, RTT, and throughput per customer are computed.
- **Top/Bottom/Frequent Values**: Lists of extreme and frequent values for TCP retransmission, RTT, and throughput are provided.
- **Visualizations**: Bar plots show the distribution of average throughput and TCP retransmission by handset type.
- **Clustering**: Users are segmented into clusters based on experience metrics, with descriptions provided for each cluster.

## Interpretation

- **Throughput and TCP Retransmission Distribution**: Provides insights into how network performance varies across different handset types.
- **Clustering**: Segments users into distinct groups based on their network experience, helping to identify areas for improvement or focus.

## Notes

- Ensure the dataset path is updated in the script.
- Review the cluster descriptions to understand the characteristics of each user group.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
