import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, request, jsonify
import joblib

etf_dir = r"path"
stock_dir = r"path"
meta_data_path = r"path"

# Read the meta data file to extract Symbol and Security Name
meta_data = pd.read_csv(meta_data_path, usecols=['Symbol', 'Security Name'])

# Create an empty dataframe to hold all the data
data = pd.DataFrame()

# Use a for loop to read each CSV file in both directories, and append it to the data dataframe
for folder, category in zip([etf_dir, stock_dir], ['ETF', 'Stock']):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        symbol = os.path.splitext(file)[0]  # extract the symbol from the file name
        df = pd.read_csv(file_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        df['Symbol'] = symbol  # add Symbol column
        if symbol in meta_data['Symbol'].values:
            df['Security Name'] = meta_data.loc[meta_data['Symbol'] == symbol, 'Security Name'].iloc[0]
        else:
            print(f"No metadata found for symbol {symbol}")
        df['Category'] = category  # add a Category column to identify ETF or Stock

        # calculate the rolling 30-day moving average of volume and rolling median of Adj Close
        df['vol_moving_avg'] = df['Volume'].rolling(window=30).mean()
        df['adj_close_rolling_med'] = df['Adj Close'].rolling(window=30).median()

        data = data._append(df, ignore_index=True)

# save the resulting dataset into a new directory
merged_path = r"path for merged file"
data.to_csv(merged_path, index=False)

new_dir = r"path"
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for symbol in data['Symbol'].unique():
    symbol_data = data[data['Symbol'] == symbol]
    symbol_path = os.path.join(new_dir, f"{symbol}.csv")
    symbol_data.to_csv(symbol_path, index=False)

# Assume `data` is loaded as a Pandas DataFrame
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Remove rows with NaN values
data.dropna(inplace=True)

# Select features and target
features = ['vol_moving_avg', 'adj_close_rolling_med']
target = 'Volume'

X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a GradientBoostingRegressor model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error and Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Save the trained model to disk
joblib.dump(model, 'gbm_model.pkl')

# Log training metrics
with open('training_metrics.txt', 'w') as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"MSE: {mse}\n")


'''
UNIT TEST
import numpy as np
def test_vol_moving_avg(data):
    # Calculate the expected moving average of trading volume for each Symbol
    expected_vol_moving_avg = data.groupby('Symbol')['Volume'].rolling(window=30).mean().reset_index(drop=True)

    # Ensure that the calculated moving average is equal to the expected value
    assert np.allclose(data['vol_moving_avg'], expected_vol_moving_avg)


def test_adj_close_rolling_med(data):
    # Calculate the expected rolling median of adjusted close for each Symbol
    expected_adj_close_rolling_med = data.groupby('Symbol')['Adj Close'].rolling(window=30).median().reset_index(drop=True)

    # Ensure that the calculated rolling median is equal to the expected value
    assert np.allclose(data['adj_close_rolling_med'], expected_adj_close_rolling_med)

'''

app = Flask(__name__)
model = joblib.load("model.pkl")  # Load the trained model

@app.route("/predict", methods=["GET"])
def predict():
    # Parse the query parameters
    vol_moving_avg = request.args.get("vol_moving_avg")
    adj_close_rolling_med = request.args.get("adj_close_rolling_med")

    # Convert the parameters to the expected data types
    vol_moving_avg = float(vol_moving_avg)
    adj_close_rolling_med = float(adj_close_rolling_med)

    # Make a prediction using the loaded model
    prediction = model.predict([[vol_moving_avg, adj_close_rolling_med]])[0]

    # Convert the prediction to an integer value and return it as a JSON response
    return jsonify(int(prediction))

if __name__ == "__main__":
    app.run()
