from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Global variables for models
linear_model = None
arima_model = None
text_generator = None  # For text generation

# Load CSV file data into a pandas DataFrame
def load_sales_data():
    df = pd.read_csv('sales_data.csv')
    return df

# Function to train the linear regression model
def train_linear_model():
    global linear_model
    df = load_sales_data()
    X = df['days'].values.reshape(-1, 1)  # Reshape for sklearn
    y = df['sales'].values
    linear_model = LinearRegression()
    linear_model.fit(X, y)

# Function to train the ARIMA model for time series analysis
def train_arima_model():
    global arima_model
    df = load_sales_data()
    sales_data = df['sales']
    arima_model = ARIMA(sales_data, order=(5, 1, 0))  # Example order (p=5, d=1, q=0)
    arima_model = arima_model.fit()

# Predict sales using the linear regression model
def predict_sales_linear(num_days):
    if linear_model is not None:
        predicted_sales = linear_model.predict(np.array([[num_days]]))
        return round(predicted_sales[0], 2)
    return None

# Predict sales using the ARIMA model for time series forecasting
def predict_sales_arima(steps):
    if arima_model is not None:
        forecast = arima_model.forecast(steps=steps)  # Forecast for 'steps' days
        # Convert forecast to list to handle multiple days of predictions
        forecast_values = forecast.tolist()
        
        # Return the first prediction if steps = 1; otherwise, return all steps
        return round(forecast_values[0], 2) if steps == 1 else [round(val, 2) for val in forecast_values]
    return None


# Function for text generation
def generate_text(prompt):
    if text_generator is not None:
        generated = text_generator(prompt, max_length=50, num_return_sequences=1)
        return generated[0]['generated_text']
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_sales_linear = None
    predicted_sales_arima = None
    generated_text = None

    if request.method == 'POST':
        num_days = request.form.get('num_days', type=int)
        steps = request.form.get('steps', type=int)  # For ARIMA prediction
        prompt = request.form.get('prompt')

        # Sales prediction using linear regression
        if num_days is not None:
            predicted_sales_linear = predict_sales_linear(num_days)

        # Sales prediction using ARIMA
        if steps is not None:
            predicted_sales_arima = predict_sales_arima(steps)

        # Text generation
        if prompt:
            generated_text = generate_text(prompt)

    return render_template('index.html', 
                           predicted_sales_linear=predicted_sales_linear,
                           predicted_sales_arima=predicted_sales_arima,
                           generated_text=generated_text)

if __name__ == '__main__':
    train_linear_model()  # Train linear regression when app starts
    train_arima_model()   # Train ARIMA model when app starts
    text_generator = pipeline('text-generation', model='gpt2')
    app.run(debug=True)
