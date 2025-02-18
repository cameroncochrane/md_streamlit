import pandas as pd
import numpy as np
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import platform

def get_data(path:str)->pd.DataFrame:
    """
    Generates a dataframe from the 'CarPrice_Assignemt.csv' file located at the given path.
    """
    data = pd.read_csv(path)

    return data

# GENERAL MANUFACTURER FUNCTIONS #
def return_manufacturer_price_metrics(data:pd.DataFrame)->pd.DataFrame:

    """
    Returns a dataframe containing the calculated price data of the manufacturers.
    """

    price_metric_names = ['Average Price','Standard Deviation','Range']

    price_metrics_data = pd.DataFrame(price_metric_names)

    for manufacturer in data['manufacturer'].unique():
        if type(manufacturer) == str: #To ignore that weired 'nan' value at the end of the manufacturers list
            manufacturer_data = data[data['manufacturer'] == manufacturer]

            average_price = manufacturer_data['price'].mean().round(2)
            sd_price = manufacturer_data['price'].std().round(2)
            range_price = manufacturer_data['price'].max() - manufacturer_data['price'].min().round(2)
            #print(f"{manufacturer}. average price: {average_price}, standard deviation: {sd_price}, range: {range_price}")
            manufacturer_entry = [average_price,sd_price,range_price]
            price_metrics_data[manufacturer] = manufacturer_entry

    price_metrics_data = price_metrics_data.transpose()
    column_names = price_metrics_data.iloc[0][:]
    price_metrics_data = price_metrics_data[:][1:]
    price_metrics_data.columns = column_names

    return price_metrics_data

def return_plot_manufacturer_averages(data:pd.DataFrame):
    """
    Uses the dataframe generated from the 'return_maunfacturer_price_metrics' function to plot the average price of all the manufacturers.
    """

    x=data.index
    y=data['Average Price']

    colors = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#FF8C33', '#33FFA8', '#A833FF',
    '#FF3333', '#33FF8C', '#A8FF33', '#5733FF', '#33A8FF', '#8C33FF', '#FFA833',
    '#33FF33', '#8CFF33', '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#FF8C33', '#33FFA8'
    ]

    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x=x,height=y,yerr=data['Standard Deviation'],color=colors)

    plt.xticks(fontsize=11,rotation=60)
    plt.yticks(fontsize=11)

    ax.set_title('Average Model Price by Manufacturer',fontsize=16,fontstyle='italic')
    ax.set_xlabel('Manufacturer',fontsize=14)
    ax.set_ylabel('Average Price ($)',fontsize=14)

    return fig

def return_plot_manufacturer_ranges(data:pd.DataFrame):
    """
    Uses the dataframe generated from the 'return_maunfacturer_price_metrics' function to plot the model price range of all the manufacturers.
    """

    x=data.index
    y=data['Range']

    colors = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#FF8C33', '#33FFA8', '#A833FF',
    '#FF3333', '#33FF8C', '#A8FF33', '#5733FF', '#33A8FF', '#8C33FF', '#FFA833',
    '#33FF33', '#8CFF33', '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#FF8C33', '#33FFA8'
    ]

    fig,ax = plt.subplots()
    ax.bar(x=x,height=y,color=colors)

    plt.xticks(fontsize=6,rotation=60)

    ax.set_title('Model Price Range by Manufacturer')
    ax.set_xlabel('Manufacturer')
    ax.set_ylabel('Price Range ($)')

    return fig

# MODEL QUANTITY + PRICES TABLE:
def return_modelq_and_prices(data:pd.DataFrame):

    manufacturer_metric_names = ['No.Models','Average Price ($)']

    manufactuer_metrics_data = pd.DataFrame(manufacturer_metric_names)

    for manufacturer in data['manufacturer'].unique():
            if type(manufacturer) == str: #To ignore that weired 'nan' value at the end of the manufacturers list
                manufacturer_data = data[data['manufacturer'] == manufacturer]
                
                average_price = manufacturer_data['price'].mean().round(2)
                model_count = manufacturer_data[manufacturer_data['manufacturer'] == manufacturer].shape[0]
                
                manufacturer_entry = [model_count,average_price]
                manufactuer_metrics_data[manufacturer] = manufacturer_entry

    manufactuer_metrics_data = manufactuer_metrics_data.transpose()
    column_names = manufactuer_metrics_data.iloc[0][:]
    manufactuer_metrics_data = manufactuer_metrics_data[:][1:]
    manufactuer_metrics_data.columns = column_names

    manufactuer_metrics_data['No.Models'] = manufactuer_metrics_data['No.Models'].astype(int) #Removing the .0 from quantity number

    return manufactuer_metrics_data
        
#ENGINE VS PRICE ML ANALYSIS FUNCTIONS #
def return_engine_ml_models(data:pd.DataFrame,metric=str):
    """
    Carries out the ML analysis of the given engine metric for the whole dataset (using sklearn LinearRegression).

    metric = 'enginesize','horsepower','city_mpg','highway_mpg'

    Returns the model, R-squared score and plot.
    """
    
    if metric == 'enginesize':

        engine_size = data['enginesize']
        price = data['price']

        x=np.array(engine_size).reshape(-1,1)
        y= price  

        enginesize_model = LinearRegression()
        enginesize_model.fit(x,y)

        y_pred = enginesize_model.predict(x)
        r_squared = r2_score(y,y_pred)

        fig,ax = plt.subplots()

        ax.scatter(engine_size,price,marker='x',c='black',label='Actual Data')
        ax.plot(x, y_pred, color='red', label='Predicted')
        ax.set_xlabel('Engine Size')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price vs Engine Size')
        ax.legend()

        return enginesize_model, r_squared, fig
    
    if metric == 'horsepower':

        horsepower = data['horsepower']
        price = data['price']

        x=np.array(horsepower).reshape(-1,1)
        y= price  

        horsepower_model = LinearRegression()
        horsepower_model.fit(x,y)

        y_pred = horsepower_model.predict(x)
        r_squared = r2_score(y,y_pred)

        fig,ax = plt.subplots()

        ax.scatter(horsepower,price,marker='x',c='black',label='Actual Data')
        ax.plot(x, y_pred, color='red', label='Predicted')
        ax.set_xlabel('Horsepower')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price vs Horsepower')
        ax.legend()

        return horsepower_model, r_squared, fig

    if metric == 'citympg':

        citympg = data['citympg']
        price = data['price']

        x=np.array(citympg).reshape(-1,1)
        y= price  

        citympg_model = LinearRegression()
        citympg_model.fit(x,y)

        y_pred = citympg_model.predict(x)
        r_squared = r2_score(y,y_pred)

        fig,ax = plt.subplots()

        ax.scatter(citympg,price,marker='x',c='black',label='Actual Data')
        ax.plot(x, y_pred, color='red', label='Predicted')
        ax.set_xlabel('City MPG')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price vs City MPG')
        ax.legend()

        return citympg_model, r_squared, fig
    
    if metric == 'highwaympg':

        highwaympg = data['highwaympg']
        price = data['price']

        x=np.array(highwaympg).reshape(-1,1)
        y= price  

        highwaympg_model = LinearRegression()
        highwaympg_model.fit(x,y)

        y_pred = highwaympg_model.predict(x)
        r_squared = r2_score(y,y_pred)

        fig,ax = plt.subplots()

        ax.scatter(highwaympg,price,marker='x',c='black',label='Actual Data')
        ax.plot(x, y_pred, color='red', label='Predicted')
        ax.set_xlabel('Highway MPG')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price vs Highway MPG')
        ax.legend()

        return highwaympg_model, r_squared, fig

# OTHER METRIC FUNCTIONS:
def model_quantity(data:pd.DataFrame):
    """
    Plots the number of different models for each manufacturer.
    """
    car_list = data['manufacturer'].unique()
    manufacturer_number_array = []

    for model in car_list:
        manufacturer_count = data[data['manufacturer'] == model].shape[0]
        manufacturer_number_array.append(manufacturer_count)

    manufacturer_quantity = {key:value for key, value in zip(car_list, manufacturer_number_array)}

    man_q_array = [manufacturer_quantity.keys(),manufacturer_quantity.values()]
    man_q_df = pd.DataFrame(data=man_q_array,index=['Car','Quantity'])
    man_q_df = man_q_df.T

    colors = [
        '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#FF8C33', '#33FFA8', '#A833FF',
        '#FF3333', '#33FF8C', '#A8FF33', '#5733FF', '#33A8FF', '#8C33FF', '#FFA833',
        '#33FF33', '#8CFF33', '#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#FF8C33', '#33FFA8'
        ]

    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x=man_q_df['Car'],height=man_q_df['Quantity'],color=colors)

    ax.set_xlabel('Manufacturer',fontsize=14)
    ax.set_ylabel('Number of Models',fontsize=14)
    ax.set_title('Model Quantity by Manufacturer',fontsize=16,fontstyle='italic')

    plt.xticks(fontsize=11,rotation=60)
    plt.yticks(fontsize=11)

    return man_q_df,fig

def fuel_type_price(data:pd.DataFrame):
    """
    Plots the price characteristics of each fuel type.
    """

    gas_df = data[data['fueltype'] == 'gas']
    diesel_df = data[data['fueltype'] == 'diesel']

    gas_average_price = gas_df['price'].mean()
    diesel_average_price = diesel_df['price'].mean()

    gas_std = gas_df['price'].std()
    diesel_std = diesel_df['price'].std()

    gas_range = gas_df['price'].max() - gas_df['price'].min()
    diesel_range = diesel_df['price'].max() - diesel_df['price'].min()
    
    fig,ax = plt.subplots()
    ax.bar(x=['Gas','Diesel'],height=[gas_average_price,diesel_average_price],width=0.5,color=['red','green'])



    ax.set_xlabel('Fuel Type',fontsize=16)
    ax.set_ylabel('Average Price ($)',fontsize=16)
    ax.set_ylim(0,20000)
    ax.set_title('Average Car Price by Fuel Type',fontsize=18,fontstyle='italic')

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13,rotation=60)

    for i, value in enumerate([round(gas_average_price,2),round(diesel_average_price,2)]):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')

    return fig

def drive_type_price(data:pd.DataFrame):
    """
    Plots the price characteristics of each drivetype.
    """

    drivewheel = ['rwd','fwd','full_wd']

    for drivetype in drivewheel:
        if drivetype == 'rwd' or drivetype == 'fwd':
            globals()[f'{drivetype}_df'] = data[data['drivewheel'] == drivetype]
        if drivetype == 'full_wd':
            globals()['full_wd_df'] = data[data['drivewheel'] == '4wd']
    
    drivetype_averages = [rwd_df['price'].mean(),fwd_df['price'].mean(),full_wd_df['price'].mean()]
    drivetype_averages = [round(num, 2) for num in drivetype_averages]

    drivetype_stds = [rwd_df['price'].std(),fwd_df['price'].std(),full_wd_df['price'].std()]
    drivetype_stds = [round(num, 2) for num in drivetype_stds]

    drivetype_ranges = [rwd_df['price'].max()-rwd_df['price'].min(),fwd_df['price'].max()-fwd_df['price'].min(),full_wd_df['price'].max()-full_wd_df['price'].min()]
    drivetype_ranges = [round(num, 2) for num in drivetype_ranges]

    drivetypes = ['RWD','FWD','4WD']
        
    fig,ax = plt.subplots()
    ax.bar(x = drivetypes, height=drivetype_averages,width=0.5,color=['red','green','blue'])
    
    ax.set_xlabel('Drivetype',fontsize=16)
    ax.set_ylabel('Average Price ($)',fontsize=16)
    ax.set_ylim(0,25000)
    ax.set_title('Average Car Price by Drivetype',fontsize=18,fontstyle='italic')

    for i, value in enumerate(drivetype_averages):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13,rotation=60)

    return fig

def engine_type_price(data:pd.DataFrame):
    """
    Plots the price characteristics of each engine type.
    """

    engine_types = data['enginetype'].unique()

    engine_type_average_prices = []
    engine_type_ranges = []
    engine_type_std = []

    for e_type in engine_types:
        average_price = data[data['enginetype'] == e_type]['price'].mean().round(2)
        engine_type_average_prices.append(average_price)

    for e_type in engine_types:
        price_range = data[data['enginetype'] == e_type]['price'].max().round(2) - data[data['enginetype'] == e_type]['price'].min().round(2)
        engine_type_ranges.append(price_range)

    for e_type in engine_types:
        price_std = data[data['enginetype'] == e_type]['price'].std().round(2)
        engine_type_std.append(price_std)

    
    engine_colors = ['#00008B', '#0000CD', '#4169E1', '#1E90FF', '#00BFFF', '#87CEFA', '#ADD8E6']

    fig,ax = plt.subplots()
    ax.bar(x=engine_types,height=engine_type_average_prices,color=engine_colors,width=0.3)

    ax.set_xlabel('Engine',fontsize=16)
    ax.set_ylabel('Average Price ($)',fontsize=16)
    ax.set_ylim(0,35000)
    ax.set_title('Average Price by Engine Type',fontsize=18,fontstyle='italic')

    for i, value in enumerate(engine_type_average_prices):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13,rotation=60)


    return fig

def fuel_system_price(data:pd.DataFrame):
    """
    Plots the price characteristics of each fuel system type.
    """

    fuel_system_types = data['fuelsystem'].unique()

    fuel_system_average_prices = []
    fuel_system_ranges = []
    fuel_system_std = []

    for f_type in fuel_system_types:
        average_price = data[data['fuelsystem'] == f_type]['price'].mean().round(2)
        fuel_system_average_prices.append(average_price)

    for f_type in fuel_system_types:
        price_range = data[data['fuelsystem'] == f_type]['price'].max().round(2) - data[data['fuelsystem'] == f_type]['price'].min().round(2)
        fuel_system_ranges.append(price_range)

    for f_type in fuel_system_types:
        price_std = data[data['fuelsystem'] == f_type]['price'].std().round(2)
        fuel_system_std.append(price_std)

    colors = ['#f54242','#f55d42','#f57842','#f59542','#f5b642','#f5cb42','#f5ec42','#e0f542']

    fig,ax = plt.subplots()
    ax.bar(x=fuel_system_types,height=fuel_system_average_prices,color=colors,width=0.3)

    ax.set_xlabel('Fuel System',fontsize=16)
    ax.set_ylabel('Average Price ($)',fontsize=16)
    ax.set_ylim(0,20000)
    ax.set_title('Average Price by Fuel System',fontsize=18,fontstyle='italic')

    for i, value in enumerate(fuel_system_average_prices):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13,rotation=60)


    return fig

def car_body_price(data:pd.DataFrame):
    """
    Plots the price characteristics of each carbody type.
    """

    car_body_types = data['carbody'].unique()

    carbody_average_prices = []
    carbody_ranges = []
    carbody_std = []

    for c_type in car_body_types:
        average_price = data[data['carbody'] == c_type]['price'].mean().round(2)
        carbody_average_prices.append(average_price)

    for c_type in car_body_types:
        price_range = data[data['carbody'] == c_type]['price'].max().round(2) - data[data['carbody'] == c_type]['price'].min().round(2)
        carbody_ranges.append(price_range)

    for c_type in car_body_types:
        price_std = data[data['carbody'] == c_type]['price'].std().round(2)
        carbody_std.append(price_std)

    body_colors = ['#006400', '#008000', '#228B22', '#32CD32', '#3CB371']

    fig,ax = plt.subplots()
    ax.bar(x=car_body_types,height=carbody_average_prices,color=body_colors,width=0.3)
    ax.xlabel('Carbody')
    ax.ylabel('Average Price ($)')
    ax.title('Average Price by Carbody')

    return fig
    
def engine_location_price(data:pd.DataFrame):
    """
    Plots the price characteristics of each engine location type.
    """

    e_location_types = data['enginelocation'].unique()

    e_location_average_prices = []
    e_location_ranges = []
    e_location_std = []

    for el_type in e_location_types:
        average_price = data[data['enginelocation'] == el_type]['price'].mean().round(2)
        e_location_average_prices.append(average_price)

    for el_type in e_location_types:
        price_range = data[data['enginelocation'] == el_type]['price'].max().round(2) - data[data['enginelocation'] == el_type]['price'].min().round(2)
        e_location_ranges.append(price_range)

    for el_type in e_location_types:
        price_std = data[data['enginelocation'] == el_type]['price'].std().round(2)
        e_location_std.append(price_std)

    fig,ax = plt.subplots()
    ax.bar(x=e_location_types,height=e_location_average_prices,color=['purple','pink'],width=0.5)

    ax.set_xlabel('Engine Location',fontsize=16)
    ax.set_ylabel('Average Price ($)',fontsize=16)
    ax.set_ylim(0,40000)
    ax.set_title('Average Price by Engine Location',fontsize=18,fontstyle='italic')

    
    for i, value in enumerate(e_location_average_prices):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13,rotation=60)


    return fig

#ENGINE VS PRICE REGRESSION (+PLOTTING) FUNCTIONS #
def horsepower_price(data:pd.DataFrame):
    """
    Plots the price vs horsepower and the regression line.
    """

    horsepower = data['horsepower']
    price = data['price']

    horsepower_price_dict = {'Horsepower':horsepower,'Price ($)':price}
    horsepower_price_df = pd.DataFrame(horsepower_price_dict)

    fig,ax = plt.subplots()
    ax.scatter(horsepower,price,marker='x',c='black', label='Actual Data')
    
    # Regression formula
    x = np.linspace(min(horsepower), max(horsepower), 100)
    y = (-0.1087 * (x ** 2)) + (191.88 * x) - 5354
    ax.plot(x, y, color='red', label='Regression Line')

    ax.set_xlabel('Horsepower')
    ax.set_ylabel('Price ($)')
    ax.legend()
    
    return horsepower_price_df, fig
    
def engine_size_price(data:pd.DataFrame):
    """
    Plots the price vs engine size and the regression line.
    """

    engine_size = data['enginesize']
    price = data['price']

    enginesize_price_dict = {'enginesize': engine_size, 'Price ($)': price}
    enginesize_price_df = pd.DataFrame(enginesize_price_dict)

    fig, ax = plt.subplots()
    ax.scatter(engine_size, price, marker='x', c='black', label='Actual Data')
    
    # Regression formula
    x = np.linspace(min(engine_size), max(engine_size), 100)
    y = (-0.0255 * (x ** 2)) + (176.23 * x) - 8633
    ax.plot(x, y, color='red', label='Regression Line')

    ax.set_xlabel('Engine Size (cc)')
    ax.set_ylabel('Price ($)')
    ax.legend()

    return enginesize_price_df, fig

def city_mpg_price(data:pd.DataFrame):
    """
    Plots the price vs city mpg and the regression line.
    """
    
    c_mpg = data['citympg']
    price = data['price']

    fig, ax = plt.subplots()
    ax.scatter(c_mpg, price, marker='x', c='black', label='Actual Data')
    
    # Regression formula
    x = np.linspace(min(c_mpg), max(c_mpg), 100)
    y = (0.149 * (x ** 4)) - (20.662 * (x ** 3)) + (1062.9 * (x ** 2)) - (24354 * x) + 219622
    ax.plot(x, y, color='red', label='Regression Line')

    ax.set_xlabel('Fuel Efficiency (MPG)')
    ax.set_ylabel('Price ($)')
    ax.legend()

    return fig

def highway_mpg_price(data:pd.DataFrame):
    """
    Plots the price vs highway mpg and the regression line.
    """
    
    h_mpg = data['highwaympg']
    price = data['price']

    fig, ax = plt.subplots()
    ax.scatter(h_mpg, price, marker='x', c='black', label='Actual Data')
    
    # Regression formula
    x = np.linspace(min(h_mpg), max(h_mpg), 100)
    y = (-0.0085 * (x ** 5)) + (1.4776 * (x ** 4)) - (100.99 * (x ** 3)) + (3433 * (x ** 2)) - (59403 * x) + 440431
    ax.plot(x, y, color='red', label='Regression Line')

    ax.set_xlabel('Fuel Efficiency (MPG)')
    ax.set_ylabel('Price ($)')
    ax.legend()

    return fig

# CAR SPECIFIC FUNCTIONS #
def manufacturer_stats(data:pd.DataFrame,manufacturer:str)->dict:
    """
    Returns a dictionary of the stats of the given manufacturer.
    """
    # Initialise the entry data and the data format of the stats to be calculated:
    m_data = data[data['manufacturer'] == manufacturer]
    m_stats = {}
    
    #Calculating the stats:
    m_stats['No. Models'] = m_data[m_data['manufacturer'] == manufacturer].shape[0]
    m_stats['Average Price ($)'] = m_data['price'].mean().round(2)
    m_stats['Price STD ($)'] = m_data['price'].std().round(2)
    m_stats['Price Range ($)'] = m_data['price'].max().round(2)-m_data['price'].min().round(2)
    m_stats['Cheapest Model ($)'] = [m_data[m_data['price'] == m_data['price'].min()]['CarName'].to_string(),m_data['price'].min().round(2)]
    m_stats['Most Expensive Model ($)'] = [m_data[m_data['price'] == m_data['price'].max()]['CarName'].to_string(),m_data['price'].max().round(2)]
    m_stats['Average Engine Size'] = m_data['enginesize'].mean().round(2)
    m_stats['Average Horsepower'] = m_data['horsepower'].mean().round(2)
    m_stats['Average Peak RPM'] = m_data['peakrpm'].mean().round(2)
    m_stats['Average City MPG'] = m_data['citympg'].mean().round(2)
    m_stats['Average Highway MPG'] = m_data['highwaympg'].mean().round(2)
    
    return m_stats

def plot_model_prices(data:pd.DataFrame,manufacturer:str):
    """
    Plots the price of each model of a given manufacturer.
    """
   
    m_data = data[data['manufacturer'] == manufacturer]

    x = list(m_data['CarName'])
    y = list(m_data['price'])
    min_y = min(y)
    max_y = max(y)

    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x,y)

    plt.xticks(fontsize=8,rotation=70)
    plt.yticks(fontsize=11)

    ax.set_title(f'{manufacturer} Model Prices', fontsize=18, fontstyle='italic')
    ax.set_xlabel('Model',fontsize=14)
    ax.set_ylabel('Price ($)',fontsize=14)
    ax.set_ylim(min_y-2000,max_y+2000)

    return fig

def plot_model_citympgs(data:pd.DataFrame,manufacturer:str):
    m_data = data[data['manufacturer'] == manufacturer]

    x = list(m_data['CarName'])
    y = list(m_data['citympg'])
    min_y = min(y)
    max_y = max(y)

   
    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x,y,color="red")

    plt.xticks(fontsize=8,rotation=70)
    plt.yticks(fontsize=11)

    ax.set_title(f'{manufacturer} City Fuel Efficiency by Model',fontsize=18, fontstyle='italic')
    ax.set_xlabel('Model',fontsize=14)
    ax.set_ylabel('Fuel Efficiency (MPG)',fontsize=14)
    ax.set_ylim(min_y-2,max_y+2)

    return fig

def plot_model_highwaympgs(data:pd.DataFrame,manufacturer:str):
    m_data = data[data['manufacturer'] == manufacturer]

    x = list(m_data['CarName'])
    y = list(m_data['highwaympg'])
    min_y = min(y)
    max_y = max(y)

   
    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x,y,color="green")

    plt.xticks(fontsize=8,rotation=70)
    plt.yticks(fontsize=11)

    ax.set_title(f'{manufacturer} Highway Fuel Efficiency by Model',fontsize=16)
    ax.set_xlabel('Model',fontsize=14)
    ax.set_ylabel('Fuel Efficiency (MPG)',fontsize=14)
    ax.set_ylim(min_y-2,max_y+2)

    return fig

def plot_model_horsepower(data:pd.DataFrame,manufacturer:str):
    m_data = data[data['manufacturer'] == manufacturer]

    x = list(m_data['CarName'])
    y = list(m_data['horsepower'])
    min_y = min(y)
    max_y = max(y)

   
    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x,y,color="yellow")

    plt.xticks(fontsize=8,rotation=70)
    plt.yticks(fontsize=11)

    ax.set_title(f'{manufacturer} Horsepower by Model',fontsize=16)
    ax.set_xlabel('Model',fontsize=14)
    ax.set_ylabel('Horsepower',fontsize=14)
    ax.set_ylim(min_y-10,max_y+10)

    return fig

#Misc:
def mpg_price(data:pd.DataFrame):
    """
    Plots the price vs mpg.
    """

    hw_mpg = data['highwaympg']
    c_mpg = data['citympg']
    price = data['price']

    fig,(ax1, ax2) = plt.subplots()
    ax1.scatter(hw_mpg,price,marker='x',c='black')
    ax1.xlabel('Highway MPG')
    ax1.ylabel('Price')

    ax2.scatter(c_mpg,price,marker='x',c='black')
    ax2.xlabel('City MPG')
    ax2.ylabel('Price')

    return fig
   
