from flask import Flask, request, render_template, send_file
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the data
data_path = 'solar_energy_data_updated.csv'
data = pd.read_csv(data_path)
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data.set_index('date', inplace=True)

# Define the SARIMAX model
model = SARIMAX(data['price in USD'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Function to generate predictions
def generate_predictions(start_date, end_date, frequency='D'):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    forecast = results.get_forecast(steps=len(date_range))
    predicted_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    predictions = pd.DataFrame({
        'Date': date_range,
        'Predicted Price in USD': predicted_mean
    })

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['price in USD'], label='Observed')
    plt.plot(date_range, predicted_mean, label='Forecast')
    plt.fill_between(date_range, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.title('Price in USD Forecast')
    plt.legend()

    # Save the plot to a file
    img_path = os.path.join('static', 'prediction_plot.png')
    plt.savefig(img_path)
    plt.close()

    return predictions, img_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    frequency = request.form.get('frequency', 'D')
    predictions, img_path = generate_predictions(start_date, end_date, frequency)

    # Calculate prediction error for the overlapping period
    historical_predictions = predictions[predictions['Date'].isin(data.index)]
    if not historical_predictions.empty:
        actuals = data.loc[historical_predictions['Date'], 'price in USD']
        prediction_error = (actuals - historical_predictions['Predicted Price in USD']).abs().mean()
    else:
        prediction_error = None

    return render_template('predictions.html', tables=[predictions.to_html(classes='data')], titles=predictions.columns.values, img_path=img_path, prediction_error=prediction_error)

@app.route('/download_data')
def download_data():
    return send_file(data_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
