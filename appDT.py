from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np

app = Flask(__name__)

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def train_model(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].map(lambda x: x.strftime('%Y-%m-%d'))
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    #data['Date'] = data['Date'].map(pd.Timestamp.timestamp)
    data['Date'] = data['Date'].apply(lambda x: pd.Timestamp(x).timestamp())

    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,train_size=0.7, random_state=0)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    return model

def predict_price(model, date):
    return model.predict(np.array([[date]]))[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        data = get_stock_data(ticker, start_date, end_date)
        model = train_model(data)

        '''
        #last_date = data.index[-1].date()
        #predicted_price = predict_price(model, pd.Timestamp.timestamp(last_date))
        '''
        
        last_date = data.index[-1].date()
        last_timestamp = pd.Timestamp(last_date).to_pydatetime().timestamp()
        predicted_price = predict_price(model, last_timestamp)

        return render_template('index.html', predicted_price=predicted_price, last_date=last_date, ticker=ticker)

    return render_template('index.html')
    
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        data = get_stock_data(ticker, start_date, end_date)
        model = train_model(data)

        last_date = data.index[-1].date()
        last_timestamp = pd.Timestamp(last_date).to_pydatetime().timestamp()
        predicted_price = predict_price(model, last_timestamp)

        # Pass the stock data, predicted price, and other relevant information to the template
        return render_template('index.html', data=data, predicted_price=predicted_price, last_date=last_date, ticker=ticker)

    return render_template('index.html')
'''

if __name__ == '__main__':
    app.run(debug=True)
