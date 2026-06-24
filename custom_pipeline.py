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
    print(" 4. 状態ごとのARIMAモデリング")
    print("==================================================")
    # 予測値を格納するカラムを準備
    df_features['sp500_pred'] = np.nan
    df_features['DGS10_pred'] = np.nan
    df_features['sp500_simulated'] = np.nan
    df_features['DGS10_simulated'] = np.nan

    # ARIMAのフィッティング
    for i in range(4):
        print(f"\n--- レジーム {i} のモデルフィッティング中 ---")
        mask = df_features['Regime'] == i
        
        # --- SPX の ARIMA モデル (exog=DGS10_lag1) ---
        y_sp500 = df_features['sp500'].copy()
        y_sp500[~mask] = np.nan
        exog_dgs = df_features['DGS10_lag1']
        
        try:
            model_sp500 = ARIMA(endog=y_sp500, exog=exog_dgs, order=(2, 1, 1))
            res_sp500 = model_sp500.fit()
            print(f"【レジーム {i}: SPX推定パラメータ】")
            print(res_sp500.params)
            
            # 予測値（フィッティング値＝条件付き期待値）を保存
            pred_mean_sp = res_sp500.predict()[mask]
            df_features.loc[mask, 'sp500_pred'] = pred_mean_sp
            sigma2_sp = res_sp500.params.get('sigma2', 0)

        except Exception as e:
            print(f"レジーム {i} のSPXモデリング中にエラーが発生しました: {e}")

        # --- DGS10 の ARIMA モデル (exog=sp500_lag1) ---
        y_dgs = df_features['DGS10'].copy()
        y_dgs[~mask] = np.nan
        exog_sp500 = df_features['sp500_lag1']
        
        try:
            model_dgs = ARIMA(endog=y_dgs, exog=exog_sp500, order=(2, 1, 1))
            res_dgs = model_dgs.fit()
            print(f"【レジーム {i}: DGS推定パラメータ】")
            print(res_dgs.params)
            
            # 予測値（フィッティング値＝条件付き期待値）を保存
            pred_mean_dgs = res_dgs.predict()[mask]
            df_features.loc[mask, 'DGS10_pred'] = pred_mean_dgs
            sigma2_dgs = res_dgs.params.get('sigma2', 1e-6)
            
            # --- DCC-GARCH モデルのフィッティング ---
            print(f"【レジーム {i}: DCC-GARCH モデルフィッティング】")
            resid_sp = res_sp500.resid[mask]
            resid_dgs = res_dgs.resid[mask]
            
            sigma2_sp = res_sp500.params.get('sigma2', 1e-6)
            
            # 標準化残差
            e_sp = resid_sp / np.sqrt(sigma2_sp)
            e_dgs = resid_dgs / np.sqrt(sigma2_dgs)
            
            # [e_T, ..., e_0] にするため逆順
            tr = np.array([e_sp[::-1], e_dgs[::-1]])
            
            dcc_model = DCC(max_itr=3)
            dcc_model.set_loss(dcc_loss_gen())
            dcc_model.fit(tr)
            ab = dcc_model.get_ab()
            print(f"最適化された DCC パラメータ (a, b) = {ab}")
            
            # 動的相関行列の生成
            R_list = R_gen(tr, ab)
            R_list_chrono = R_list[::-1]
            
            std_sp = np.sqrt(sigma2_sp)
            std_dgs = np.sqrt(sigma2_dgs)
            
            shocks_sp = np.zeros(mask.sum())
            shocks_dgs = np.zeros(mask.sum())
            
            for t in range(mask.sum()):
                R_t = R_list_chrono[t]
                rho_t = R_t[0, 1]
                cov_t = rho_t * std_sp * std_dgs
                cov_matrix_t = [[sigma2_sp, cov_t], [cov_t, sigma2_dgs]]
                
                # シードは固定しないか、必要に応じて固定（ここではランダムに）
                shock_t = np.random.multivariate_normal([0, 0], cov_matrix_t)
                shocks_sp[t] = shock_t[0]
                shocks_dgs[t] = shock_t[1]
            
            df_features.loc[mask, 'sp500_simulated'] = pred_mean_sp + shocks_sp
            df_features.loc[mask, 'DGS10_simulated'] = pred_mean_dgs + shocks_dgs

        except Exception as e:
            print(f"レジーム {i} のDGSまたはDCCモデリング中にエラーが発生しました: {e}")

    print("\n==================================================")
    print(" 5. モデルによる出力（実際の値と予測値・シミュレーション値の比較）")
    print("==================================================")
    
    # 最後の出力をする際に、小数点以下6桁に丸める
    round_cols = ['sp500_pred', 'sp500_simulated', 'DGS10_pred', 'DGS10_simulated']
    df_features[round_cols] = df_features[round_cols].round(6)
    
    output_df = df_features[['Regime', 'sp500', 'sp500_pred', 'sp500_simulated', 'DGS10', 'DGS10_pred', 'DGS10_simulated']]
    print("【直近10日間のデータ（実際の値・予測値・シミュレーション値）】")
    print(output_df.tail(10))
    
    output_path = '/home/u00118/Analysing_Models(sonomamadata)/Marcov_Switching_Model/predictions_output.csv'
    output_df.to_csv(output_path)
    print(f"\n-> 予測結果全体をCSVに保存しました: {output_path}")

    print("\n==================================================")
    print(" 6. 全期間での累積リターンの比較プロット生成")
    print("==================================================")
    # 累積リターンの計算
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
    print(" 7. 実データとモデル生成データの相関比較 (1-Year Rolling)")
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
