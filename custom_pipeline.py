import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.arima.model import ARIMA
import warnings
import sys
import os

repo_path = os.path.join(os.getcwd(), 'DCC_GARCH_repo')
if repo_path not in sys.path:
    sys.path.append(repo_path)
    
try:
    from DCC_GARCH.DCC.DCC import DCC
    from DCC_GARCH.DCC.DCC_loss import dcc_loss_gen, R_gen
    from DCC_GARCH.GARCH.GARCH import GARCH
    from DCC_GARCH.GARCH.GARCH_loss import garch_loss_gen
except ImportError as e:
    print("DCC-GARCH repository import failed:", e)

warnings.filterwarnings('ignore')

def main():
    print("==================================================")
    print(" 1. データの読み込みと特徴量エンジニアリング")
    print("==================================================")
    df = pd.read_csv('/home/u00118/train_sp500_us10y.csv')
    if 'Unnamed: 0' in df.columns:
        df['Date'] = pd.to_datetime(df['Unnamed: 0'])
        df.set_index('Date', inplace=True)
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # ローリング特徴量の作成 (20日 = 約1ヶ月)
    window = 20
    df['sp500_roll_mean'] = df['sp500'].rolling(window=window).mean()
    df['DGS10_roll_mean'] = df['DGS10'].rolling(window=window).mean()
    df['sp500_roll_std'] = df['sp500'].rolling(window=window).std()
    df['DGS10_roll_std'] = df['DGS10'].rolling(window=window).std()
    df['corr'] = df['sp500'].rolling(window=window).corr(df['DGS10'])

    # もう一方の変数の過去1日分のデータ（外生変数用）
    df['DGS10_lag1'] = df['DGS10'].shift(1)
    df['sp500_lag1'] = df['sp500'].shift(1)

    # 欠損値の削除 (最初の19日分など)
    df_features = df.dropna().copy()
    print(f"有効なデータ件数: {len(df_features)}件")

    features = df_features[['sp500_roll_mean', 'DGS10_roll_mean', 'sp500_roll_std', 'DGS10_roll_std', 'corr']]

    print("\n==================================================")
    print(" 2. クラスタリング（GMMを用いた4レジームへの分類）")
    print("==================================================")
    # 乱数シードを固定してGMMクラスタリング
    gmm = GaussianMixture(n_components=4, random_state=42, n_init=10)
    df_features['Regime'] = gmm.fit_predict(features)
    
    print("各レジームのデータ件数:")
    print(df_features['Regime'].value_counts().sort_index())

    # プロットの保存
    plt.figure(figsize=(15, 6))
    plt.plot(df_features.index, df_features['sp500'], color='black', alpha=0.3, label='S&P 500 Return')
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        mask = df_features['Regime'] == i
        plt.scatter(df_features.index[mask], df_features['sp500'][mask], color=colors[i], label=f'Regime {i}', s=10)
    plt.title('S&P 500 Returns by GMM Clustered Regime')
    plt.legend()
    plot_path = '/home/u00118/Analysing_Models(sonomamadata)/Marcov_Switching_Model/regime_plot.png'
    plt.savefig(plot_path)
    print(f"-> プロットを保存しました: {plot_path}")

    print("\n==================================================")
    print(" 3. 状態遷移確率行列の計算")
    print("==================================================")
    transition_matrix = np.zeros((4, 4))
    regimes = df_features['Regime'].values
    for t in range(1, len(regimes)):
        from_state = regimes[t-1]
        to_state = regimes[t]
        transition_matrix[from_state, to_state] += 1

    # 行ごとに正規化して確率に変換
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    print("【推移確率行列 (Transition Matrix)】")
    for i in range(4):
        print(f"状態 {i} からの遷移確率: [0]:{transition_matrix[i,0]:.3f}, [1]:{transition_matrix[i,1]:.3f}, [2]:{transition_matrix[i,2]:.3f}, [3]:{transition_matrix[i,3]:.3f}")

    print("\n==================================================")
    print(" 4. 状態ごとのARIMAモデリング（期待値の計算）")
    print("==================================================")
    # 予測値を格納するカラムを準備
    df_features['sp500_pred'] = np.nan
    df_features['DGS10_pred'] = np.nan

    # ARIMAのフィッティング
    for i in range(4):
        print(f"\n--- レジーム {i} のARIMAモデリング中 ---")
        mask = df_features['Regime'] == i
        
        # --- SPX の ARIMA モデル ---
        y_sp500 = df_features['sp500'].copy()
        y_sp500[~mask] = np.nan
        exog_dgs = df_features['DGS10_lag1']
        
        try:
            model_sp500 = ARIMA(endog=y_sp500, exog=exog_dgs, order=(2, 1, 1))
            res_sp500 = model_sp500.fit()
            pred_mean_sp = res_sp500.predict()[mask]
            df_features.loc[mask, 'sp500_pred'] = pred_mean_sp
        except Exception as e:
            print(f"レジーム {i} のSPXモデリング中にエラー: {e}")

        # --- DGS10 の ARIMA モデル ---
        y_dgs = df_features['DGS10'].copy()
        y_dgs[~mask] = np.nan
        exog_sp500 = df_features['sp500_lag1']
        
        try:
            model_dgs = ARIMA(endog=y_dgs, exog=exog_sp500, order=(2, 1, 1))
            res_dgs = model_dgs.fit()
            pred_mean_dgs = res_dgs.predict()[mask]
            df_features.loc[mask, 'DGS10_pred'] = pred_mean_dgs
        except Exception as e:
            print(f"レジーム {i} のDGSモデリング中にエラー: {e}")

    # 残差（予測誤差）を全期間連続データとして計算
    df_features['resid_sp500'] = df_features['sp500'] - df_features['sp500_pred']
    df_features['resid_dgs'] = df_features['DGS10'] - df_features['DGS10_pred']
    
    # 稀にNaNが含まれる場合は0埋め（通常はない）
    df_features['resid_sp500'].fillna(0, inplace=True)
    df_features['resid_dgs'].fillna(0, inplace=True)

    print("\n==================================================")
    print(" 5. 全期間を通した GARCH および DCC-GARCH モデリング")
    print("==================================================")
    
    resid_sp = df_features['resid_sp500'].values
    resid_dgs = df_features['resid_dgs'].values
    
    # リポジトリの仕様 [r_T, ..., r_0] の順にするため逆順にする。
    # GARCHの最適化（COBYLA等）が収束しやすいように、スケールを100倍（%表記）にする
    r_sp_rev = resid_sp[::-1] * 100.0
    r_dgs_rev = resid_dgs[::-1] * 100.0

    print("【SPX: GARCH(1,1) モデルフィッティング】")
    garch_sp = GARCH(p=1, q=1, max_itr=3)
    garch_sp.set_loss(garch_loss_gen(p=1, q=1))
    garch_sp.fit(r_sp_rev)
    # sigma() は [s_T, ..., s_0] を返すので逆順にして [s_0, ..., s_T] に戻す
    sigma_sp = garch_sp.sigma(r_sp_rev)[::-1]
    
    print("【DGS10: GARCH(1,1) モデルフィッティング】")
    garch_dgs = GARCH(p=1, q=1, max_itr=3)
    garch_dgs.set_loss(garch_loss_gen(1, 1))
    garch_dgs.fit(r_dgs_rev)
    sigma_dgs = garch_dgs.sigma(r_dgs_rev)[::-1]
    
    # 標準化残差 (スケールを合わせたまま割り算)
    e_sp = (resid_sp * 100.0) / sigma_sp
    e_dgs = (resid_dgs * 100.0) / sigma_dgs
    
    # DCCフィッティング用データ [e_T, ..., e_0]
    tr = np.array([e_sp[::-1], e_dgs[::-1]])
    
    # FHS用: 各レジームごとの標準化残差プール作成
    regime_noise_pools = {}
    for i in range(4):
        mask = df_features['Regime'] == i
        regime_noise_pools[i] = np.array([e_sp[mask], e_dgs[mask]]).T
    
    print("【DCC-GARCH モデルフィッティング】")
    dcc_model = DCC(max_itr=3)
    dcc_model.set_loss(dcc_loss_gen())
    dcc_model.fit(tr)
    ab = dcc_model.get_ab()
    print(f"最適化された DCC パラメータ (a, b) = {ab}")
    
    # シミュレーションの実行
    print("--- 動的分散と動的相関を用いたレジーム依存・再帰的シミュレーションの実行 (FHS) ---")
    df_features['sp500_simulated'] = np.nan
    df_features['DGS10_simulated'] = np.nan
    shocks_sp = np.zeros(len(df_features))
    shocks_dgs = np.zeros(len(df_features))
    
    # 再帰的シミュレーションのための初期化
    theta_sp = garch_sp.get_theta()
    theta_dgs = garch_dgs.get_theta()
    
    s_sp_t = sigma_sp[0]
    s_dgs_t = sigma_dgs[0]
    
    ab_dcc = dcc_model.get_ab()
    a_dcc, b_dcc = ab_dcc[0], ab_dcc[1]
    Q_bar = Q_average(tr)
    Q_t = Q_bar.copy()
    
    for t in range(len(df_features)):
        # 1. 今日の相関 (rho_t) の計算
        temp = 1.0 / np.sqrt(np.abs(Q_t))
        temp = temp * np.eye(2)
        R_t = np.dot(np.dot(temp, Q_t), temp)
        rho_t = R_t[0, 1]
        
        if rho_t**2 >= 1.0:
            rho_t = np.sign(rho_t) * 0.999
        
        # GARCHで100倍したスケールを元に戻す
        sigma_sp_orig = s_sp_t / 100.0
        sigma_dgs_orig = s_dgs_t / 100.0
        
        # シミュレーション時点のレジーム
        sim_regime = df_features['Regime'].iloc[t]
        
        # レジームプールからランダムに過去の独立ノイズ(z_tau)を1つ引く
        pool = regime_noise_pools[sim_regime]
        z_idx = np.random.randint(len(pool))
        z_sim = pool[z_idx]
        
        # シミュレーション時点の相関(L_t)を掛けて再相関化
        L_t = np.array([
            [1.0, 0.0],
            [rho_t, np.sqrt(max(1.0 - rho_t**2, 1e-6))]
        ])
        e_sim = L_t @ z_sim
        
        # シミュレーション時点のGARCHボラティリティを掛ける
        shocks_sp[t] = e_sim[0] * sigma_sp_orig
        shocks_dgs[t] = e_sim[1] * sigma_dgs_orig
        
        # ----------------------------------------------------
        # 明日のための再帰的アップデート (Recursive Update)
        # ----------------------------------------------------
        # GARCHの更新 (100倍スケールのショックを利用)
        r_sp_t_scaled = shocks_sp[t] * 100.0
        r_dgs_t_scaled = shocks_dgs[t] * 100.0
        
        var_sp = s_sp_t ** 2
        r_sq_sp = r_sp_t_scaled ** 2
        gjr_sp = r_sq_sp * (r_sp_t_scaled < 0)
        s_sp_t = np.sqrt(np.abs(theta_sp[0] + theta_sp[1]*r_sq_sp + theta_sp[2]*gjr_sp + theta_sp[3]*var_sp))
        
        var_dgs = s_dgs_t ** 2
        r_sq_dgs = r_dgs_t_scaled ** 2
        gjr_dgs = r_sq_dgs * (r_dgs_t_scaled < 0)
        s_dgs_t = np.sqrt(np.abs(theta_dgs[0] + theta_dgs[1]*r_sq_dgs + theta_dgs[2]*gjr_dgs + theta_dgs[3]*var_dgs))
        
        # DCCの更新
        Q_t = (1.0 - a_dcc - b_dcc) * Q_bar + a_dcc * np.outer(e_sim, e_sim) + b_dcc * Q_t
        
    df_features['sp500_simulated'] = df_features['sp500_pred'] + shocks_sp
    df_features['DGS10_simulated'] = df_features['DGS10_pred'] + shocks_dgs

    print("\n==================================================")
    print(" 6. モデルによる出力（実際の値と予測値・シミュレーション値の比較）")
    print("==================================================")
    
    round_cols = ['sp500_pred', 'sp500_simulated', 'DGS10_pred', 'DGS10_simulated']
    df_features[round_cols] = df_features[round_cols].round(6)
    
    output_df = df_features[['Regime', 'sp500', 'sp500_pred', 'sp500_simulated', 'DGS10', 'DGS10_pred', 'DGS10_simulated']]
    print("【直近10日間のデータ】")
    print(output_df.tail(10))
    
    output_path = '/home/u00118/Analysing_Models(sonomamadata)/Marcov_Switching_Model/predictions_output.csv'
    output_df.to_csv(output_path)
    print(f"\n-> 予測結果全体をCSVに保存しました: {output_path}")

    print("\n==================================================")
    print(" 7. 全期間での累積リターンの比較プロット生成")
    print("==================================================")
    df_features['cum_sp500'] = (1 + df_features['sp500'].fillna(0)).cumprod()
    df_features['cum_sp500_pred'] = (1 + df_features['sp500_pred'].fillna(0)).cumprod()
    df_features['cum_sp500_sim'] = (1 + df_features['sp500_simulated'].fillna(0)).cumprod()

    plt.figure(figsize=(15, 6))
    plt.plot(df_features.index, df_features['cum_sp500'], label='Actual SPX Cumulative Return', color='black')
    plt.plot(df_features.index, df_features['cum_sp500_pred'], label='Predicted SPX Cumulative Return', color='blue', linestyle='--')
    plt.plot(df_features.index, df_features['cum_sp500_sim'], label='Simulated SPX Cumulative Return', color='red', alpha=0.7)
    plt.title('Cumulative Returns: Actual vs Predicted vs Simulated (Initial Value = 1)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    cum_plot_path = '/home/u00118/Analysing_Models(sonomamadata)/Marcov_Switching_Model/cumulative_returns_plot.png'
    plt.savefig(cum_plot_path)
    plt.close()
    print(f"-> 累積リターンのプロットを保存しました: {cum_plot_path}")

    print("\n==================================================")
    print(" 8. 実データとモデル生成データの相関比較 (1-Year Rolling)")
    print("==================================================")
    df_features['1Y_Label'] = df_features.index.year.astype(str)
    
    corr_act = df_features.groupby('1Y_Label').apply(lambda g: g['sp500'].corr(g['DGS10']))
    corr_pred = df_features.groupby('1Y_Label').apply(lambda g: g['sp500_pred'].corr(g['DGS10_pred']))
    corr_sim = df_features.groupby('1Y_Label').apply(lambda g: g['sp500_simulated'].corr(g['DGS10_simulated']))
    
    corr_1y = pd.DataFrame({'Actual': corr_act, 'Predicted': corr_pred, 'Simulated': corr_sim})
    
    plt.figure(figsize=(20, 6))
    plt.plot(corr_1y.index, corr_1y['Actual'], marker='o', label='Actual Data', color='black', linewidth=1.5, markersize=4)
    plt.plot(corr_1y.index, corr_1y['Predicted'], marker='x', label='Predicted (Mean)', color='blue', linestyle='--', linewidth=1.5, markersize=4)
    plt.plot(corr_1y.index, corr_1y['Simulated'], marker='s', label='Simulated (with shocks)', color='red', alpha=0.7, linewidth=1.5, markersize=4)
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    plt.title('1-Year Rolling Correlation between SPX and DGS10 (Actual vs Predicted vs Simulated)')
    plt.xlabel('Year')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    corr_plot_path = '/home/u00118/Analysing_Models(sonomamadata)/Marcov_Switching_Model/correlation_1y_plot.png'
    plt.savefig(corr_plot_path)
    plt.close()
    print(f"-> 1年ごとの相関係数推移プロットを保存しました: {corr_plot_path}")

    vol_act = corr_1y['Actual'].std()
    vol_pred = corr_1y['Predicted'].std()
    vol_sim = corr_1y['Simulated'].std()
    
    print("\n【相関係数のボラティリティ（標準偏差）の比較】")
    print(f"1. 実データの相関のボラティリティ            : {vol_act:.4f}")
    print(f"2. 予測データ(期待値)の相関のボラティリティ  : {vol_pred:.4f}")
    print(f"3. シミュレーションの相関のボラティリティ    : {vol_sim:.4f}")

if __name__ == "__main__":
    main()
