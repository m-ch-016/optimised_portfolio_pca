import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov


tickers = ["INTC", "ORLA", "RTX", "CMG", "XOM"]
data = yf.download(tickers, start="2022-01-01", end="2024-01-01", auto_adjust=False)

adj_close = data['Adj Close']

returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252 
cov_matrix = returns.cov() * 252    

mu = mean_historical_return(data)
S = sample_cov(data)

ef = EfficientFrontier(mu, S)

weights = ef.max_sharpe() 
cleaned_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)