import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np 
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
merged_path = r"path"
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

data_x=X
data_y=y

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'lr': LinearRegression(),
    'lasso': Lasso(alpha=0.1),
    'ridge': Ridge(alpha=0.1),
    'rf': RandomForestRegressor(n_estimators=100),
    'svm':SVR(kernel='rbf', gamma='auto')
    }


def load_historic_data(start_date, end_date):
    # define an empty list to store the loaded data
    data = []

    # loop over each date in the range of start_date and end_date
    # assuming each date has its own set of csv files
    for date in pd.date_range(start=start_date, end=end_date, freq='D'):
        # construct the file path for the csv file corresponding to the current date
        file_path = f"path/to/csv/files/{date.strftime('%Y-%m-%d')}.csv"

        # load the data from the csv file
        # assuming the csv file has columns 'x' and 'y'
        df = pd.read_csv(file_path, usecols=['x', 'y'])

        # append the loaded data to the list
        data.append(df)

    # concatenate the loaded data into a single dataframe
    data = pd.concat(data, axis=0)

    return data

# Load the historic data from 2023-1-1 till current date
historic_data = load_historic_data(start_date='2023-1-1', end_date='2023-5-2')

# assuming that your data is stored in a pandas dataframe called 'data'
start_date = '2023-01-01'
end_date = '2023-05-02'

# subset the data between the start and end dates
subset_data = data.loc[start_date:end_date]

# create the X and y variables for each model
X = subset_data.drop('target', axis=1)
y = subset_data['target']

# initialize the retrogressively_fitted_weights dictionary
retrogressively_fitted_weights = {}

# loop over each model and calculate the retrogressively_fitted_weights
for model_name, model in models.items():
    # fit the model retrogressively to get the weights
    model.fit(X[::-1], y[::-1])
    # get the coefficients (weights) of the fitted model
    weights = model.coef_
    # add the retrogressively fitted weights to the dictionary
    retrogressively_fitted_weights[model_name] = weights

# Initialize weights with random values
weights = retrogressively_fitted_weights

# Define the learning rate
learning_rate = 0.01

# Perform retrogressive gradient descent on the historic data
for i in range(historic_data.shape[0] - 1, 0, -1):
    # Get the inputs and outputs for this timestep
    X = historic_data[i - 1:i, :-1]
    y = historic_data[i:i + 1, -1].reshape(-1, 1)

    # Compute the predicted value
    y_pred = np.dot(X, weights)

    # Compute the error
    error = y_pred - y

    # Update the weights using gradient descent
    weights -= learning_rate * np.dot(X.T, error)


#Calculate MSE AND MAE
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
def mean_actual_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Loop until accuracy threshold is met
dates = []
start_date = datetime.datetime(2022, 1, 1)
for i in range(10):
    date = start_date + datetime.timedelta(days=i)
    dates.append(date)

for i in range(len(dates)):
    train_x = data_x[:i]
    train_y = data_y[:i]
    test_x = data_x[i:i + 1]
    test_y = data_y[i:i + 1]

    # Train models on train set
    for name, model in models.items():
        model.fit(train_x, train_y)

    # Predict on test set and calculate errors
    errors = {}
    for name, model in models.items():
        pred_y = model.predict(test_x)[0]
        errors[name] = mean_actual_error(test_y, pred_y)

    # Update weights based on errors
    total_error = sum(errors.values())
    for name, error in errors.items():
        if error > 0:
            weights[name] *= (total_error / error)

    # Normalize weights
    weight_sum = sum(weights.values())
    for name in weights:
        weights[name] /= weight_sum

    # Predict volumes using weighted average of models
    pred_volume = 0
    for name, model in models.items():
        pred_volume += weights[name] * model.predict(test_x)[0]

    # Print results
    print(f"Date: {dates[i]}, Actual Volume: {test_y[0]}, Predicted Volume: {pred_volume}")

# save the retrogressively fitted weights to a file using joblib
joblib.dump(retrogressively_fitted_weights, 'model.pkl')
