import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv('train_sp500_us10y.csv')
df.dropna(inplace=True)
ret_sp = df['sp500'].values
ret_dgs = df['DGS10'].values

# 2. Simple log-likelihood for DCC
def dcc_loss(params, z1, z2):
    a, b = params
    if a <= 0 or b <= 0 or a + b >= 1:
        return 1e10 # constraint penalty
    
    T = len(z1)
    Q_bar = np.cov(z1, z2)
    Q = Q_bar.copy()
    
    ll = 0
    for t in range(T):
        # Current correlation
        R11 = 1.0 # Q[0,0] / sqrt(Q[0,0]*Q[0,0])
        R22 = 1.0
        R12 = Q[0,1] / np.sqrt(Q[0,0] * Q[1,1])
        
        # det and inv
        det_R = 1 - R12**2
        if det_R <= 0.0001:
            det_R = 0.0001
        
        inv_R_11 = 1 / det_R
        inv_R_22 = 1 / det_R
        inv_R_12 = -R12 / det_R
        
        z_t = np.array([z1[t], z2[t]])
        
        # Likelihood contribution
        ll += np.log(det_R) + (z_t[0]**2 * inv_R_11 + z_t[1]**2 * inv_R_22 + 2*z_t[0]*z_t[1] * inv_R_12)
        
        # Update Q
        Q = (1 - a - b) * Q_bar + a * np.outer(z_t, z_t) + b * Q
        
    return 0.5 * ll

print("Optimizing DCC...")
res = minimize(dcc_loss, [0.05, 0.90], args=(ret_sp[:1000]/np.std(ret_sp[:1000]), ret_dgs[:1000]/np.std(ret_dgs[:1000])), bounds=[(0.001, 0.999), (0.001, 0.999)])
print(res)

