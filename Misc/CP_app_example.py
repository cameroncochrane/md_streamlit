## App: https://cp-data-analysis.streamlit.app/ 

#Streamlit Modules
import streamlit as st
from streamlit_option_menu import option_menu

#Base Modules:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#For showing images:
from PIL import Image

#Manipulating strings:
import re

#Custom Modules:
import car_price_functions as cp

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Initialising the format
st.set_page_config(layout="wide")

#Setting up the data
path = 'data/CarPrice_Assignment.csv'
data = cp.get_data(path=path)
manufacturer_list = list(data['manufacturer'].unique())

#Pages
def home_page():
    st.title("Car Price Market  Analysis")

    st.write("This app is an analysis of the new car market. The app is formatted in a way that allows us to easily view the analysis")
    
    st.write("""
    **Instructions:**
    - **Home:** Overview of the application and disclaimer.
    - **Overview:** General data overview including manufacturer model quantity and price, and general car features.
    - **Engine and Price Analysis:** Analysis of car prices based on various engine metrics such as fuel efficiency, horsepower, and engine size.
    - **Manufacturer Pages:** Detailed analysis for each manufacturer, including general stats and visualizations of model prices, fuel efficiency, and horsepower.
    """)

    st.write("""
    **Disclaimer:** The data used in this analysis is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Automobile) and falls under the CC BY 4.0 license. 
    This dataset is open and publicly available. Any dataset submitted to Libertas Data for analysis will not be used to showcase our products/services, and will not be made open and publicly available under any circumstances, ensuring data protection and anonymity. 
    This analysis represents the quality of service that our company offers.
    """)

def dashboard_page():
    st.title("Overview")
    
    average_prices = cp.return_manufacturer_price_metrics(data=data)
 
    col_1 = st.columns((10,10), gap='small')

    with col_1[0]:
        model_number_df, model_number_plot, = cp.model_quantity(data=data)
        st.pyplot(fig=model_number_plot,use_container_width=True)

        st.header("Manufactuer Model Quantity and Price")

        model_quant_and_prices_df = cp.return_modelq_and_prices(data)
        st.dataframe(data=model_quant_and_prices_df,width=400, height=900)

    with col_1[1]:

        average_prices_plot = cp.return_plot_manufacturer_averages(data=average_prices)
        st.pyplot(fig=average_prices_plot,use_container_width=True)

        st.header("General Car Features")

        col_2 = st.columns((5,5), gap='small')

        with col_2[0]:

            fuel_type_plot = cp.fuel_type_price(data)
            st.pyplot(fig=fuel_type_plot,use_container_width=True)

            engine_type_plot = cp.engine_type_price(data)
            st.pyplot(fig=engine_type_plot,use_container_width=True)

            fuel_system_type_plot = cp.fuel_system_price(data)
            st.pyplot(fig=fuel_system_type_plot,use_container_width=True)


        with col_2[1]:

            drive_type_plot = cp.drive_type_price(data)
            st.pyplot(fig=drive_type_plot,use_container_width=True)

            engine_location_type = cp.engine_location_price(data)
            st.pyplot(fig=engine_location_type,use_container_width=True)

            carbody_type_plot = cp.drive_type_price(data)
            st.pyplot(fig=carbody_type_plot,use_container_width=True)
        
        st.subheader("Abbreviations Key")

        col_3 = st.columns((1,1), gap='small')

        with col_3[0]:
            st.write("**Fuel Systems**")
            st.write("""
            - **1bbl**: Single-barrel carburetor
            - **2bbl**: Two-barrel carburetor
            - **4bbl**: Four-barrel carburetor
            - **idi**: Indirect Diesel Injection
            - **mfi**: Multiport Fuel Injection
            - **mpfi**: Multi-Point Fuel Injection
            - **spdi**: Single-Point Direct Injection
            - **spfi**: Single-Point Fuel Injection
            """)

        with col_3[1]:
            st.write("**Engine Types**")
            st.write("""
            - **dohc**: Double Overhead Camshaft
            - **ohcv**: Overhead Valve
            - **ohc**: Overhead Camshaft
            - **I**: Inline Engine
            - **rotor**: Rotary Engine
            - **ohcf**: Overhead Camshaft with Fuel Injection
            - **dohcv**: Double Overhead Camshaft with Variable Valve Timing
            """)

def engine_metrics_vs_price_page():

    ## NEED TO ADD THE GRAPHS NEXT ##
    st.title("Engine Metrics and Price Analysis")

    st.write("Use the sliders below each graph to predict the price of a car based on different engine metrics.")


    st.subheader(":blue[Fuel Efficiency]")

    col_1 = st.columns((1,1), gap='medium')
    with col_1[0]:
        

        #City MPG Visualisation + Predictions
        st.write("**City Fuel Efficiency**")

        c_mpg_price_plot = cp.city_mpg_price(data)
        st.pyplot(fig=c_mpg_price_plot,use_container_width=True)

        #Define the input city MPG value, determined by the slide and stored as a integer in a variable:
        city_mpg_value = st.slider(label="",
                                min_value=13,
                                max_value=50,
                                value=13,
                                step=1)
        
        def predict_price_city_MPG(city_MPG:int)->float:
            """"
            Predicts the price of a car based on the city MPG value. Regression formula (determined in Excel) used:
            y = (0.149*(x)^4)-(20.662*(x)^3)+(1062.9*(x)^2)-(24354*x)+219622
            """
            return (0.149*(city_MPG)**4)-(20.662*(city_MPG)**3)+(1062.9*(city_MPG)**2)-(24354*city_MPG)+219622
        
        st.write(f"Predicted Price: ${predict_price_city_MPG(city_mpg_value):,.2f}")

    with col_1[1]:
        #Highway MPG Visualisation + Predictions
        st.write("**Highway Fuel Efficiency**")

        h_mpg_price_plot = cp.highway_mpg_price(data)
        st.pyplot(fig=h_mpg_price_plot,use_container_width=True)

        highway_mpg_value = st.slider(label="",
                                    min_value=16,
                                    max_value=50,
                                    value=16,
                                    step=1)
        
        def predict_price_highway_MPG(highway_MPG:int)->float:
            """
            Predicts the price of a car based on the highway MPG value. Regression formula (determined in Excel) used:
            y = (-0.0085*(x)^5)+(1.4776*(x)^4)-(100.99*(x)^3)+(3433*(x)^2)-(59403*x)+440431
            """
            return (-0.0085*(highway_MPG)**5)+(1.4776*(highway_MPG)**4)-(100.99*(highway_MPG)**3)+(3433*(highway_MPG)**2)-(59403*highway_MPG)+440431
        
        st.write(f"Predicted Price: ${predict_price_highway_MPG(highway_mpg_value):,.2f}")

    col_2 = st.columns((1,1), gap='medium')
    with col_2[0]:
        # Horsepower Visualisation + Predictions
        st.subheader(":red[Horsepower]")

        horsepower_data,horsepower_plot = cp.horsepower_price(data)
        st.pyplot(fig=horsepower_plot,use_container_width=True)

        horsepower_value = st.slider(label="",
                                    min_value=48,
                                    max_value=288,
                                    value=48,
                                    step=1)
        
        def predict_price_horsepower(horsepower:int)->float:
            """
            Predicts the price of a car based on the horsepower value. Regression formula (determined in Excel) used:
            y =(-0.1087*(x)^2)+(191.88*x)-5354
            """
            return (-0.1087*(horsepower)**2)+(191.88*horsepower)-5354

        st.write(f"Predicted Price: ${predict_price_horsepower(horsepower_value):,.2f}")
    
    with col_2[1]:
        # Engine Size Visualisation + Predictions
        st.subheader(":green[Engine Size]")

        engine_size_data,engine_size_plot = cp.engine_size_price(data)
        st.pyplot(fig=engine_size_plot,use_container_width=True)

        engine_size_value = st.slider(label="",
                                    min_value=61,
                                    max_value=326,
                                    value=61,
                                    step=1)
        
        def predict_price_engine_size(engine_size:int)->float:
            """
            Predicts the price of a car based on the engine size value. Regression formula (determined in Excel) used:
            y=(-0.0255*(x)^2)+(176.23*x)-8633
            """
            return (-0.0255*(engine_size)**2)+(176.23*engine_size)-8633

        st.write(f"Predicted Price: ${predict_price_engine_size(engine_size_value):,.2f}")

def manufacturer_page(manufacturer:str)->None:

    st.title(f"{manufacturer}")
    
    col_1 = st.columns((1,2), gap='medium')

    with col_1[0]:
        #Getting the general stats of the given manufacturer:
        man_stats = cp.manufacturer_stats(data=data,manufacturer=manufacturer)

        # Opening the logo dynamically based on the selected manufacturer:
        image_path = f"car_logos/{manufacturer}.png"
        image = Image.open(image_path)
        st.image(image,use_container_width=True)
        st.divider()

        st.header('General Stats')
        st.markdown("")

        col_2 = st.columns((2,1),gap='small')

        with col_2[0]:

            st.write('No. of Different Models:')
            st.write('Average Price ($)')
            st.write('Price Range ($)')
            st.write('Price Standard Deviation ($).')
            st.write('Average Horespower')
            st.write('Average Engine Size (cc)')

            #See below regarding the removal of the starting digits of the model names for most/least expensive models:
            st.write('Most Expensive Model')
            st.write('Least Expensive Model')

        with col_2[1]:

            st.write(man_stats['No. Models'])
            st.write(man_stats['Average Price ($)'])
            st.write(man_stats['Price Range ($)'])
            st.write(man_stats['Price STD ($)'])
            st.write(man_stats['Average Horsepower'])
            st.write(man_stats['Average Engine Size'])

            #Formatting: removing the starting digits of the model names for most/least expensive models:
            def remove_leading_number(s: str) -> str:
                """
                Removes a leading 2- or 3-digit number (and the space that follows)
                from a string, if it exists.
                """
                # This regex matches:
                #   ^      Start of the string
                #   \d{2,3}  2 or 3 digits
                #   \s+    One or more whitespace characters
                return re.sub(r'^\d{2,3}\s+', '', s)
            
            formatted_most_expensive = remove_leading_number(man_stats['Most Expensive Model ($)'][0])
            formatted_cheapest = remove_leading_number(man_stats['Cheapest Model ($)'][0])

            #Outputting the most and least expensive models (formatted):
            st.write(formatted_most_expensive)
            st.write(formatted_cheapest)
        
    with col_1[1]:
        
        model_prices = cp.plot_model_prices(data=data,manufacturer=manufacturer)
        st.pyplot(fig=model_prices,use_container_width=True)

        model_cmpgs = cp.plot_model_citympgs(data=data,manufacturer=manufacturer)
        st.pyplot(fig=model_cmpgs,use_container_width=True)

        model_hmpgs = cp.plot_model_highwaympgs(data=data,manufacturer=manufacturer)
        st.pyplot(fig=model_hmpgs,use_container_width=True)

        model_horsepower = cp.plot_model_horsepower(data=data,manufacturer=manufacturer)
        st.pyplot(fig=model_horsepower,use_container_width=True)

        ## Find a way to dynamically change the minimum y_lim so that there is more resolution on the bar charts (making bars easier to distinguish). (DONE)
        ##...and the color. (DONE)
        ## Graphs seem quite big, can we make them smaller (divide into columns?)
     
#CSS styling of the sidebar + Adding the sidebar, and its behaviour:   
menu_styling_css ={
        "container": {"padding": "25px 50px 75px 100px", "background-color": "#f7f7f7"},
        "icon": {"color": "#627cd1", "font-size": "25px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#6f7178"},
        "nav-link-selected": {"background-color": "#5f78b8"},
}

with st.sidebar:
    selected = option_menu(menu_title="Main Menu",
                           options = ["Home","Overview","Engine and Price Analysis"] + manufacturer_list,
                           icons=["house","zoom-in","car-front-fill"], #Bootsdrap Icons: https://icons.getbootstrap.com/ 
                           menu_icon="list",
                           default_index=0,
                           styles=menu_styling_css #See the css dictionary above)
    )

if selected == "Home":
    home_page()

if selected == "Overview":
    dashboard_page()

if selected == "Engine and Price Analysis":
    engine_metrics_vs_price_page()
    ## NEEDS TO BE WORKED ON  (NOTHING ADDED YET) ##
    ## Need to work out how sliders work

for manufacturer in manufacturer_list:
    if selected == manufacturer:
        manufacturer_page(manufacturer)

        