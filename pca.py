import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

tks = ["INTC", "ORLA", "RTX", "CMG", "XOM", "GOOG"]
raw = yf.download(tks, start="2022-01-01", end="2024-01-01", auto_adjust=False)
prices = raw['Adj Close']
rets = prices.pct_change().dropna()

mu = mean_historical_return(prices)
cov = sample_cov(prices)

ef = EfficientFrontier(mu, cov, weight_bounds=(0.05, 0.6))
ef.min_volatility()

w = ef.clean_weights()

exp_ret, vol, sr = ef.portfolio_performance()

pca = PCA()
pca.fit(rets)

var = pca.explained_variance_ratio_

load = pd.DataFrame(pca.components_, columns=rets.columns)
load.index = [f"PC{i+1}" for i in range(load.shape[0])]

fac = rets.dot(load.T)

etfs = ['XLE', 'XLK', 'XLF', 'SPY', 'VTI', 'QQQ']

etf_px = yf.download(etfs, start='2022-01-01', end='2024-01-01', auto_adjust=True)['Close']
etf_rets = etf_px.pct_change().dropna()

fac_a, etf_a = fac.align(etf_rets, join='inner', axis=0)

corr = pd.DataFrame(index=fac_a.columns, columns=etf_a.columns)

for pc in fac_a.columns:
    for e in etf_a.columns:
        corr.loc[pc, e] = fac_a[pc].corr(etf_a[e])

corr = corr.iloc[:4, :]

fig = plt.figure(figsize=(20, 14))
fig.suptitle("PCA + Portfolio Analysis", fontsize=22, weight='bold', y=0.97)

ax1 = fig.add_subplot(2, 2, 1)

x = range(1, len(var) + 1)
ax1.plot(x, var, marker='o')

for i, v in enumerate(var):
    ax1.text(x[i], v + 0.002, f"{v:.2f}", ha='center', fontsize=9)

ax1.set_title("Scree Plot + PCA Loadings")
ax1.set_xlabel("PC")
ax1.set_ylabel("Variance")
ax1.grid(True)

text = (
    f"Sharpe: {sr:.2f}\n"
    f"ExpReturn: {exp_ret:.2%}\n\n"
    "Weights:\n" +
    "\n".join([f"{k}: {v:.2%}" for k, v in w.items()])
)

ax1.text(0.98, 0.98, text, transform=ax1.transAxes,
         fontsize=10, va='top', ha='right',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


tbl = ax1.table(cellText=load.round(2).values,
                colLabels=load.columns.tolist(),
                rowLabels=load.index.tolist(),
                loc='bottom',
                bbox=[0, -0.65, 1, 0.3])

tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

ax1.text(0.5, -0.3, "PCA Loadings", fontsize=12,
         ha='center', va='center', transform=ax1.transAxes)

ax2 = fig.add_subplot(2, 2, 2)

for i in range(min(3, fac.shape[1])):
    ax2.plot(fac.index, fac.iloc[:, i], label=f'PC{i+1}')

ax2.set_title("Factor Returns (PC1–PC3)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Return")
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(2, 2, 3)
ax3.axis('off')
interp = (
    "Based on the heatmap:\n"
    "- PC1 ~ broad market\n"
    "- PC2 ~ tech\n"
    "- PC3 ~ energy"
)
ax3.text(0, 0.8, interp, fontsize=12, va='top')

ax4 = fig.add_axes([0.57, 0.08, 0.37, 0.32])
sns.heatmap(corr.astype(float), annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)

ax4.set_title("PCA vs Sector ETFs")
ax4.set_ylabel("Principal Components")
ax4.set_xlabel("ETFs")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
