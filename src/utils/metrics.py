import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-6):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def smape(y_true, y_pred, eps=1e-6):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

def evaluate_all(y_true, y_pred):
    return {
        "MAE": round(float(mae(y_true, y_pred)), 5),
        "RMSE": round(float(rmse(y_true, y_pred)), 5),
        "MAPE": round(float(mape(y_true, y_pred)), 3),
        "sMAPE": round(float(smape(y_true, y_pred)), 3),
    }