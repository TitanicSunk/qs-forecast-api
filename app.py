from flask import Flask, jsonify, send_file
from datetime import datetime, timedelta
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Welcome to the QS Forecast API</h1><p>Try <a href='/api/qs-chart'>/api/qs-chart</a> or <a href='/api/qs-trend'>/api/qs-trend</a></p>"
# -------------------------------
# API Endpoint 1: JSON forecast
# -------------------------------
@app.route("/api/qs-trend", methods=["GET"])
def qs_trend():
    try:
        # Define date range
        end_date = datetime.today()
        start_date = end_date - timedelta(days=730)  # last 2 years

        # Download QS data
        df = yf.download("QS", start=start_date, end=end_date, progress=False)[['Close']]
        df.reset_index(inplace=True)
        df.columns = ['ds', 'y']

        # Forecast using Prophet
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Return last 30 forecasted results
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
        return jsonify(result.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# API Endpoint 2: Forecast chart
# -------------------------------
@app.route("/api/qs-chart", methods=["GET"])
def qs_chart():
    try:
        # Define date range
        end_date = datetime.today()
        start_date = end_date - timedelta(days=730)

        # Download QS data
        df = yf.download("QS", start=start_date, end=end_date, progress=False)[['Close']]
        df.reset_index(inplace=True)
        df.columns = ['ds', 'y']

        # Forecast using Prophet
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['ds'], df['y'], 'k.', label='Historical Price')
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        color='skyblue', alpha=0.4, label='Confidence Interval')

        # Annotate chart
        ax.axvline(x=datetime.today(), color='gray', linestyle='--', alpha=0.6)
        ax.text(datetime.today(), ax.get_ylim()[1]*0.95, 'Today', rotation=90, color='gray')
        ax.set_title("QS Stock Price Forecast (Next 30 Days)", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Closing Price (USD)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        # Serve as PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Run Flask server
# -------------------------------
if __name__ == "__main__":
    import os

port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port, debug=True)



