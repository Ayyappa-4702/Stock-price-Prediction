from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def train_model_with_cv(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].map(lambda x: x.strftime('%Y-%m-%d'))
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    #['Date'] = data['Date'].map(pd.Timestamp.timestamp)
    data['Date'] = data['Date'].apply(lambda x: pd.Timestamp(x).timestamp())

    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values

    model = LinearRegression()

    # Use TimeSeriesSplit for time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')

    # Fit the model on the entire dataset
    model.fit(X, y)

    return model, cv_scores

def predict_price(model, date):
    return model.predict(np.array([[date]]))[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        data = get_stock_data(ticker, start_date, end_date)
        model, cv_scores = train_model_with_cv(data)

        last_date = data.index[-1].date()
        last_timestamp = pd.Timestamp(last_date).to_pydatetime().timestamp()
        predicted_price = predict_price(model, last_timestamp)

        return render_template('index.html', predicted_price=predicted_price, last_date=last_date, ticker=ticker, cv_scores=cv_scores)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
