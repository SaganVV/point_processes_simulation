import numpy as np
import plotly.express as px

def cummean(arr):
    return np.cumsum(arr) / (1 + np.arange(len(arr)))

def acf(X, mean_estimate=None, nlags=30):
    X = np.asarray(X)
    if mean_estimate is None:
        mean_estimate = np.mean(X)
    var = np.cov(X)
    n = len(X)
    X_centered = X - mean_estimate
    corr = [1]
    for lag in range(1, nlags+1):
        est_cov = X_centered[:n-lag] @ X_centered[lag:] / n
        corr.append(est_cov / var)
    return np.asarray(corr)

def acf_plot(X, nlags=30):

    corr = acf(X, nlags=nlags)

    fig = px.scatter(y=corr)
    fig.update_layout(
        yaxis_range=[-1.2, 1.2],
        title="Autocorrelation of number of points",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation"
    )
    fig.add_hline(y=0)
    return fig

def mcmc_mean_variance_estimator(samples, acf_nlags = None):
    if acf_nlags is None:
        acf_nlags = len(samples) // 50
    corr = acf(samples, nlags=acf_nlags)
    var = np.cov(samples)
    return var * (1 + 2*np.sum(corr[1:])) / len(samples)

if __name__ ==  "__main__":
    X = np.arange(50)
    print(X)
    print(acf(X, nlags=10))
    fig = acf_plot(X, nlags=10)
    fig.show()