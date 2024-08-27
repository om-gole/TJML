#Om Gole, Period 6, Gabor
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Function to find the best ARIMA model parameters
def find_best_arima_params(time_series):
    auto_model = auto_arima(time_series, start_p=1, start_q=1, max_p=5, max_q=5, seasonal=False, trace=True)
    return auto_model.order

# Function to fit ARIMA model and forecast
def fit_arima_and_forecast(time_series, order, steps=5):
    model = ARIMA(time_series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return model_fit, forecast

# Function to plot historical data and forecast
def plot_stock_data(time_series, forecast):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Historical Closing Prices', 'Forecasted Prices'))
    
    # Historical data plot
    fig.add_trace(go.Scatter(x=time_series.index, y=time_series, name='Historical'), row=1, col=1)
    
    # Forecast plot
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast'), row=2, col=1)
    
    fig.update_layout(height=600, width=800, title_text='Stock Data Analysis')
    fig.show()

# Example usage
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'
time_series = fetch_stock_data(ticker, start_date, end_date)
order = find_best_arima_params(time_series)
model_fit, forecast = fit_arima_and_forecast(time_series, order)
plot_stock_data(time_series, forecast)

# Temporal Difference Learning Example (Advanced)
# Assuming a custom environment and policy (not shown here for brevity)
# This would involve creating a custom environment class and a policy function
# The policy function would take the current state and return an action
# The environment class would have methods to reset the environment, step through it, and render it
# The TD learning code would interact with this environment using the policy


