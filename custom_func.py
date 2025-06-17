import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

def plot_scatter_with_stats(dataframe, outlier, feature) -> None:
    index = dataframe.index
    data = dataframe[feature]
    
    mean_value = data.mean()
    std_dev = data.std()
    lower_bound = mean_value - 3 * std_dev
    upper_bound = mean_value + 3 * std_dev

    outliers = (data < lower_bound) | (data > upper_bound)   

    feature_data = outlier.loc[feature]
    if isinstance(feature_data, DataFrame):
        for i in range(len(feature_data)):
            print(f"Outlier in feature \'{feature}\' at index {int(feature_data.iloc[i]['at_index'])}: {feature_data.iloc[i]['value']} (z-score: {(feature_data.iloc[i]['value']-mean_value)/std_dev:.4f})")
    else: 
        print(f"Outlier in feature \'{feature}\' at index {int(feature_data['at_index'])}: {feature_data['value']} (z-score: {(feature_data['value']-mean_value)/std_dev:.4f})")
            
    plt.figure(figsize=(10, 6))  

    plt.scatter(index[~outliers], data[~outliers], label=feature, color='blue', marker='o', s=50)

    if outliers.any():
        plt.scatter(index[outliers], data[outliers], label=f'{feature} (outliers)', color='red', marker='x', s=100)

    plt.axhline(y=mean_value, color='orange', linestyle='-', linewidth=2, label=f'mean {feature}')

    plt.axhline(y=lower_bound, color='g', linestyle='--', linewidth=2, label=f'-3 standard deviation')
    plt.axhline(y=upper_bound, color='g', linestyle='--', linewidth=2, label=f'+3 standard deviation')

    plt.title(f'Plot of {feature} values')
    plt.xlabel('Index of data point')
    plt.ylabel(feature)
    plt.grid(True)
    plt.legend()
    plt.show()


def correl_heat_map(corr_matrix) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, linewidths=.5, fmt=".2f")
    plt.title('Correlation heatmap', fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    