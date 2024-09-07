# Exploratory Data Analysis (EDA) for User Behavior Analysis

This project focuses on performing exploratory data analysis (EDA) on user behavior data across multiple applications. The objectives include aggregating user session information, identifying patterns, and deriving useful insights by analyzing data usage across various applications.

## Table of Contents

- [Task 1.1](#task-11)
  - [Objective](#objective-11)
  - [Steps Performed](#steps-performed-11)
  - [Results](#results-11)
- [Task 1.2](#task-12)
  - [Objective](#objective-12)
  - [Steps Performed](#steps-performed-12)
  - [Results](#results-12)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Task 1.1

### Objective

To aggregate and analyze user data, including:

- Session duration per user.
- Total download (DL) and upload (UL) data.
- Total data volume (in Bytes) per session for each application.

### Steps Performed

1. **Data Aggregation**: Calculated total session duration, total download, and upload data per user.
2. **Data Cleaning**: Identified and handled missing values and outliers using appropriate techniques (e.g., replacing with the mean).
3. **Descriptive Analysis**: Described the relevant variables and their data types.
4. **Variable Analysis**: Segmented users into decile classes based on total session duration and computed total data per decile.

### Results

- Successfully aggregated data to provide an overview of user behavior.
- Identified patterns and trends in user data consumption across various applications.

## Task 1.2

### Objective

To conduct an in-depth exploratory data analysis on the aggregated data from Task 1.1, including:

- Identifying and treating missing values and outliers.
- Performing variable transformations and univariate/bivariate analysis.
- Computing correlation and conducting dimensionality reduction using Principal Component Analysis (PCA).

### Steps Performed

1. **Data Cleaning**:
   - Treated missing values and outliers by replacing them with appropriate values (e.g., mean).
2. **Variable Transformations**:
   - Segmented users into the top five decile classes based on total session duration.
   - Computed the total data usage (DL + UL) per decile class.
3. **Univariate Analysis**:
   - Conducted both non-graphical (e.g., mean, median) and graphical analysis for all quantitative variables.
4. **Bivariate Analysis**:
   - Explored the relationships between data usage per application and the total data (DL + UL).
5. **Correlation Analysis**:
   - Computed the correlation matrix for different data types (e.g., social media, YouTube, Netflix).
6. **Dimensionality Reduction**:
   - Performed PCA to reduce the dimensions of the dataset and interpret the most significant components.

### Results

- Uncovered key insights into user behavior across multiple applications.
- Identified correlations between different types of data usage.
- Reduced data dimensions to highlight the most impactful variables.

## Installation

To run this project, you need to have Python installed with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
