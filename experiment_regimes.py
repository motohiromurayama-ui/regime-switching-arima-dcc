import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.metrics import mean_squared_error
import os

warnings.filterwarnings('ignore')

# データの読み込み
df = pd.read_csv('/home/u00118/train_sp500_us10y.csv')
if 'Unnamed: 0' in df.columns:
    df['Date'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Date', inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)

# ローリング特徴量の作成 (20日 = 約1ヶ月)
window = 20
df['sp500_roll_std'] = df['sp500'].rolling(window=window).std()
df['DGS10_roll_std'] = df['DGS10'].rolling(window=window).std()
df['corr'] = df['sp500'].rolling(window=window).corr(df['DGS10'])

# 20日間の変化分（トレンド）の追加
df['sp500_roll_ret20'] = (1 + df['sp500']).rolling(window=window).apply(np.prod, raw=True) - 1
df['DGS10_roll_sum20'] = df['DGS10'].rolling(window=window).sum()

# もう一方の変数の過去1日分のデータ（外生変数用）
df['DGS10_lag1'] = df['DGS10'].shift(1)
df['sp500_lag1'] = df['sp500'].shift(1)

# 欠損値の削除
df_features = df.dropna().copy()

# GMMクラスタリング用の特徴量
features = df_features[['sp500_roll_std', 'DGS10_roll_std', 'corr', 'sp500_roll_ret20', 'DGS10_roll_sum20']].values

actual_returns = df_features['sp500']
actual_cum = (1 + actual_returns).cumprod()

results = []

for n in range(10, 21):
    print(f"Testing N_REGIMES = {n}")
    hmm_model = GaussianHMM(n_components=n, covariance_type="full", random_state=42, n_iter=100)
    hmm_model.fit(features)
    regimes = hmm_model.predict(features)
    
    pred_returns = pd.Series(np.nan, index=df_features.index)
    
    for i in range(n):
        mask = (regimes == i)
        if sum(mask) < 20:
            continue
            
        y_sp500 = df_features['sp500'].copy()
        y_sp500[~mask] = np.nan
        exog_dgs = df_features['DGS10_lag1']
        
        try:
            model_sp500 = ARIMA(endog=y_sp500, exog=exog_dgs, order=(2, 0, 2))
            res_sp500 = model_sp500.fit()
            pred_returns[mask] = res_sp500.predict()[mask]
        except Exception as e:
            # Convergence errors or others
            pass
            
    # Fill remaining NaNs (e.g. from failed models or very small regimes) with 0 return
    pred_returns = pred_returns.fillna(0)
    pred_cum = (1 + pred_returns).cumprod()
    
    # Calculate difference
    mse = mean_squared_error(actual_cum, pred_cum)
    results.append({
        'N_REGIMES': n,
        'MSE': mse,
        'Final_Actual': actual_cum.iloc[-1],
        'Final_Pred': pred_cum.iloc[-1]
    })
    print(f"  MSE: {mse:.4f}, Final Actual: {actual_cum.iloc[-1]:.4f}, Final Pred: {pred_cum.iloc[-1]:.4f}")

df_results = pd.DataFrame(results)
print("\nResults Summary:")
print(df_results.sort_values('MSE'))
df_results.to_csv('experiment_results.csv', index=False)
