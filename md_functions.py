import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(path:str)->pd.DataFrame:
    """
    Generates a dataframe from the 'superstore_data.csv' file located at the given path.
    """
    data = pd.read_csv(path)

    return data

#Data manipulation functions:

def include_ts_bs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds total monetary spent and total number of purchases columns to the dataframe (ts).

    Also adds the brackets/bins columns (bs) for the income and birthyear columns.

    Birthyear bins are as follows: '1940-1949', '1950-1959', '1960-1969', '1970-1979', '1980-1989', '1990-1999'

    Income bins are as follows: '0-9999', '10000-19999', '20000-29999', '30000-39999', '40000-49999', '50000-59999', '60000-69999', '70000-79999', '80000-89999', '90000-99999', '100000+'

    Parameters:
    data (pd.DataFrame): The input dataframe containing columns for individual monetary spends and purchases.

    Returns:
    pd.DataFrame: The dataframe with additional columns: 'TotalMntSpent', 'TotalNumPurchases', 'Birthyear_Bin', and 'Income_Bin'.
    """

    #Totals columns:
    data['TotalMntSpent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
    data['TotalNumPurchases'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']  

    #Bin columns:
    income_bins = [0, 9999, 19999, 29999, 39999, 49999, 59999, 69999, 79999, 89999, 99999, np.inf]
    income_labels = ['0-9999', '10000-19999', '20000-29999', '30000-39999', '40000-49999', '50000-59999', '60000-69999', '70000-79999', '80000-89999', '90000-99999', '100000+']
    data['Income_Bin'] = pd.cut(data['Income'], bins=income_bins, labels=income_labels, right=False)

    birthyear_bins = [1940, 1950, 1960, 1970, 1980, 1990, 2000]
    birthyear_labels = ['1940-1949', '1950-1959', '1960-1969', '1970-1979', '1980-1989', '1990-1999']
    data['Birthyear_Bin'] = pd.cut(data['Year_Birth'], bins=birthyear_bins, labels=birthyear_labels, right=False)

    return data


def filter_level(data:pd.DataFrame,metric:str)->dict:
    """
    Filters the dataframe based on the unique level and returns a dictionary with dataframes for each unique level in a set of levels of a given metric e.g different education levels.

    Parameters:
    data (pd.DataFrame): The input dataframe containing the data from the get_data() and include_totals() functions.

    metric (str): The column name in the dataframe to filter by. Can be 'Education','Marital_Status','Birthyear_Bin' or 'Income_Bin'.
    
    Returns:
    dict: A dictionary where the keys are the unique levels and the values are dataframes filtered by each unique level.
    """
    def return_level(data:pd.DataFrame, metric:str,level:str):
        """
        metric can be 'Education','Marital_Status','Birthyear_Bin' or 'Income_Bin'.
        """
        return data[data[str(metric)] == level]
    
    unique_levels = data[metric].unique()

    data_by_metric = {level:return_level(data,metric,level) for level in unique_levels}

    return data_by_metric


def calculate_metrics(data:dict)->dict:
    """
    Calculates various metrics for each unique level in the provided data.

    Can be used for education level, marital status, income bracket and birthyear (EMIB).

    Parameters:
    data (dict): A dictionary where the keys are different levels and the values are dataframes filtered by those unique levels.

    Returns:
    dict: A dictionary where the keys are unique levels and the values are dictionaries containing calculated metrics.
          The metrics include:
          - avg_total_mnt_spent: Average total monetary spent.
          - std_total_mnt_spent: Standard deviation of total monetary spent.
          - range_total_mnt_spent: Range (max - min) of total monetary spent.
          - avg_total_num_purchases: Average total number of purchases.
          - std_total_num_purchases: Standard deviation of total number of purchases.
          - range_total_num_purchases: Range (max - min) of total number of purchases.
          - response_rate: Response rate as a percentage.
          - no_customers: Number of customers.
          - sum_purchases: Sum of total number of purchases.
    """
    data_metrics = {}

    for level, df in data.items():
        avg_total_mnt_spent = round(df['TotalMntSpent'].mean(), 1)
        std_total_mnt_spent = round(df['TotalMntSpent'].std(), 1)
        range_total_mnt_spent = round(df['TotalMntSpent'].max() - df['TotalMntSpent'].min(), 1)
        
        avg_total_num_purchases = round(df['TotalNumPurchases'].mean(), 1)
        std_total_num_purchases = round(df['TotalNumPurchases'].std(), 1)
        range_total_num_purchases = round(df['TotalNumPurchases'].max() - df['TotalNumPurchases'].min(), 0)
        
        response_rate = round((df['Response'].sum() / len(df)) * 100, 1)
        no_customers = len(df)
        sum_purchases = df['TotalNumPurchases'].sum()
        
        data_metrics[level] = {
            'avg_total_mnt_spent': avg_total_mnt_spent,
            'std_total_mnt_spent': std_total_mnt_spent,
            'range_total_mnt_spent': range_total_mnt_spent,
            'avg_total_num_purchases': avg_total_num_purchases,
            'std_total_num_purchases': std_total_num_purchases,
            'sum_purchases': sum_purchases,
            'range_total_num_purchases': range_total_num_purchases,
            'response_rate': response_rate,
            'no_customers': no_customers
        }
    
    return data_metrics


def product_type_stats(data: dict)->dict:
    """
    data: dict (of pd.DataFrame) generated from the 'filter_level' function.
    
    Returns a dictionary with the total amount spent, average amount spent, and number of customers for each product category.

    """
    categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    stats = {}
    for category, df in data.items():
        total_spent = df[categories].sum()
        average_spent = df[categories].mean()
        no_customers = len(df)
        stats[category] = {
            'total_spent': total_spent,
            'average_spent': average_spent,
            'no_customers': no_customers
        }
    return stats

def extract_metric(metrics: dict, metric_name: str) -> dict:
    """
    Extracts a specific metric from the dictionary of metrics (generated from the 'calculate_metrics' function).

    Parameters:
    metrics (dict): A dictionary of metrics.
    metric_name (str): The name of the metric to extract. Can be one of the following:

    'avg_total_mnt_spent', 'std_total_mnt_spent', 'range_total_mnt_spent', 'avg_total_num_purchases', 'std_total_num_purchases', 'sum_purchases', 'range_total_num_purchases', 'response_rate', 'no_customers'

    Returns:
    dict: A dictionary containing the metric for each level.
    """
    
    return {level: metrics[level][metric_name] for level in metrics}


#Plotting functions

def plot_customer_spend_and_number(averages: dict, customers: dict):
    """
    Plots the average spend and number of customers for each level for the EMIB analysis.

    Parameters:
    averages (dict): A dictionary containing the average spend per customer for each level of a given feature.
    customers (dict): A dictionary containing the number of customers for each level of a given feature.

    Returns:
    matplotlib.figure.Figure: The plot figure.
    """
    fig, ax1 = plt.subplots()

    category_levels = list(averages.keys())
    avg_spend = list(averages.values())
    num_customers = list(customers.values())

    ax1.bar(category_levels, avg_spend, color='b', alpha=0.6, label='Average Spend')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Average Spend', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(category_levels, num_customers, color='r', marker='o', label='Number of Customers')
    ax2.set_ylabel('Number of Customers', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    return fig


def plot_marketing_response(response_rates: dict) -> plt.Figure:
    """
    Plots the marketing response rates for each category level.

    Parameters:
    response_rates (dict): A dictionary where the keys are category levels (e.g., education levels) and the values are the response rates for each level.

    Returns:
    matplotlib.figure.Figure: The plot figure showing the response rates as a horizontal bar chart.
    
    """
    
    fig, ax1 = plt.subplots()

    category_levels = list(response_rates.keys())
    response_rate = list(response_rates.values())
    
    ax1.barh(category_levels, response_rate, color='g', alpha=0.6, label='Response Rate')
    ax1.set_xlabel('Response Rate', color='g')
    ax1.set_ylabel('Category')
    ax1.tick_params(axis='x', labelcolor='g')

    fig.tight_layout()
    return fig


def plot_customer_average_purchase_quantity_and_number(purchase_quantity: dict, customers: dict):
    """
    Plots the average purchase quantity and number of customers for each category level.

    Parameters:
    purchase_quantity (dict): A dictionary where the keys are category levels (e.g., education levels) and the values are the average purchase quantities for each level.
    customers (dict): A dictionary where the keys are category levels and the values are the number of customers for each level.

    Returns:
    matplotlib.figure.Figure: The plot figure showing the average purchase quantities as a bar chart and the number of customers as a line chart.
    """
    
    fig, ax1 = plt.subplots()

    category_levels = list(purchase_quantity.keys())
    purchases = list(purchase_quantity.values())
    num_customers = list(customers.values())

    ax1.bar(category_levels, purchases, color='grey', alpha=0.6, label='Average Spend')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Average Purchase Quantity', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(category_levels, num_customers, color='r', marker='o', label='Number of Customers')
    ax2.set_ylabel('Number of Customers', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    return fig


def extract_average_spent_per_product(category_products: dict) -> dict:
    """
    Extracts the average spent for each product type for each category level.

    Parameters:
    category_products (dict): A dictionary where the keys are the category levels and the values are dictionaries containing product spending statistics.

    Returns:
    dict: A dictionary where the keys are category levels and the values are dictionaries containing the average spent for each product type.
    """
    average_spent_per_product = {}
    
    for level, stats in category_products.items():
        average_spent_per_product[level] = stats['average_spent'].to_dict()
    
    return average_spent_per_product

def plot_average_customer_spend_products(average_spent_per_product: dict):
    """
    Plots a stacked bar chart where the bars (for each level) are made up of a combination of the average spent for each product.

    Parameters:
    average_spent_per_product (dict): A dictionary where the keys are category levels and the values are dictionaries containing the average spent for each product type.
    (generated from the 'extract_average_spent_per_product' function)

    Returns:
    matplotlib.figure.Figure: The plot figure showing the stacked bar chart.
    """
    fig, ax = plt.subplots()

    category_levels = list(average_spent_per_product.keys())
    product_types = list(next(iter(average_spent_per_product.values())).keys())

    # Initialize bottom to zero for stacking
    bottom = np.zeros(len(category_levels))

    colors = {
        'MntWines': 'darkblue',
        'MntFruits': 'orange',
        'MntMeatProducts': 'darkgreen',
        'MntFishProducts': 'lightblue',
        'MntSweetProducts': 'violet',
        'MntGoldProds': 'green'
    }

    for product in product_types:
        values = [average_spent_per_product[level][product] for level in category_levels]
        ax.bar(category_levels, values, bottom=bottom, label=product, color=colors[product])
        bottom += values

    ax.set_xlabel('Category Level')
    ax.set_ylabel('Average Spend')
    ax.set_title('Average Customer Spend per Product by Category Level')
    ax.legend(title='Product Type')

    fig.tight_layout()
    return fig
