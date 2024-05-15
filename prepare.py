import pandas as pd
import os

def get_data(directory, files_ext = '.csv', sep=','):
    files = [f for f in os.listdir(directory) if files_ext in f]
    
    titles = []
    datasets = []

    for file in files:
        titles.append(file.split('.')[0])
        dataset = pd.read_csv(directory + file, sep=sep)
        datasets.append(dataset)

    return dict(zip(titles, datasets))

def get_dataset(time_series:pd.Series, window_size):
    values = list(time_series)
    n = len(values)
    
    X = []
    Y = []
    
    for i in range(n - window_size):
        X.append(values[i:i + window_size])
        Y.append(values[i + window_size])
        
    return np.array(X), np.array(Y)

def get_prediction(model, periods, initial_values:pd.Series, reshape=0):
    x_pred = [list(initial_values)]
    predicts = []
    
    for i in range(periods):
        if reshape:
            pred = model.predict(np.array([[x_pred[i]]]).reshape((1, len(list(initial_values)), 1)), verbose=0)[0]
        else:
            pred = model.predict(np.array([x_pred[i]]))
        predicts.append(pred)
        new_x = x_pred[i][1:]
        new_x.append(pred[0])
        x_pred.append(new_x)
        
    return predicts

def print_metrics(train, test, pred_train, pred_test):
    print("================== Train ==================")
    print("R^2 = ", r2_score(train, pred_train))
    print("MSE = ", mean_squared_error(train, pred_train))
    
    print("\n================== Test ===================")
    print("MSE = ", mean_squared_error(test, pred_test))