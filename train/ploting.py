import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_feature_importance(model_xgb, X_train):
    # Plot feature importance
    feature_importance = model_xgb.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def plot_train_val_metric(ytest, ypred):
    x_ax = range(len(ytest))
    plt.plot(x_ax, ytest, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("test data and predicted data")
    plt.legend()
    plt.show()