import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV  # <--- 改用 RidgeCV 自動交叉驗證
import re
import warnings
import time

warnings.filterwarnings('ignore')

# =========================
# CONFIG (參數設定)
# =========================
# Ridge 的懲罰強度候選列表 (讓 CV 自動從這裡面選)
RIDGE_ALPHA_LIST = np.arange(0.1, 5, 0.1)

MIN_PRICE = 10       # 最低股價
NW_LAGS = 4          # Newey-West Lags

# =========================
# 1. 基礎工具
# =========================
def to_month_end(s):
    s_str = s.astype(str)
    d1 = pd.to_datetime(s_str, format='%Y%m', errors='coerce')
    d2 = pd.to_datetime(s_str, format='%Y-%m', errors='coerce')
    d3 = pd.to_datetime(s_str, errors='coerce')
    d = d1.fillna(d2).fillna(d3)
    return d.dt.to_period('M').dt.to_timestamp('M')

def extract_code_token(s: str):
    s = str(s).strip().upper()
    if not s: return ""
    tok = s.split()[0].split('-')[0]
    return re.sub(r'[^0-9A-Z]', '', tok)

def to_quarter_end(s):
    s = pd.Series(s)
    return pd.to_datetime(s, errors='coerce').dt.to_period('Q').dt.to_timestamp('Q')

# =========================
# 2. 資料前處理
# =========================
def prepare_fund_data(df_fund, code_col='證券代碼', date_col='年月', ret_col='單月ROI', pct_as_percent=True):
    df = df_fund.copy()
    df[code_col] = df[code_col].astype(str).str.strip()
    df[date_col] = to_month_end(df[date_col])
    df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
    if pct_as_percent: df[ret_col] /= 100
    return df[[code_col, date_col, ret_col]]

def prep_holding_from_fund_data(fund_data, fund_col='證券代碼', q_col='年季', stock_id='標的碼', weight_col='投資比率％', keep_asset_col='投資標的', keep_asset_value='股票型'):
    h = fund_data.copy()
    if keep_asset_col in h.columns:
        h = h[h[keep_asset_col].astype(str).str.contains(str(keep_asset_value), na=False)]
    h['基金代碼'] = h[fund_col].astype(str).str.strip()
    h['年季'] = pd.PeriodIndex(h[q_col].astype(str).str.strip(), freq='Q')
    h['key_code'] = h[stock_id].astype(str).apply(extract_code_token)
    h['w_raw'] = pd.to_numeric(h[weight_col], errors='coerce') / 100
    h = h.groupby(['基金代碼', '年季', 'key_code'], as_index=False).agg(w_raw=('w_raw', 'sum'))
    h = h[h['w_raw'] > 0].copy()
    h['w'] = h['w_raw'] / h.groupby(['基金代碼', '年季'])['w_raw'].transform('sum')
    h['w'] = h['w'].fillna(0)
    return h[['基金代碼', '年季', 'key_code', 'w']]

# =========================
# 3. 核心運算: 機器學習預測 (RidgeCV Version)
# =========================
def run_ml_stock_prediction_ridge_cv(holding_w, stock_q, ridge_alphas, min_funds=10, lookback_quarters=2):
    """
    使用 RidgeCV 進行「滾動視窗交叉驗證 (Rolling Window CV)」。
    
    邏輯：
    1. 針對每一個 Target Q (t)，取過去 lookback_quarters 季 (t-lookback ... t-1) 作為訓練資料。
    2. 在訓練資料內部，自動進行 5-Fold Cross Validation (切分股票) 找出最佳 Alpha。
    3. 用最佳 Alpha 訓練完後，預測 Target Q。
    """
    
    # 1. 資料合併
    merged = pd.merge(holding_w, stock_q, on=['key_code', '年季'], how='inner')
    unique_quarters = sorted(merged['年季'].unique())
    print(f"      > ML RidgeCV (Rolling Lookback={lookback_quarters}, 5-Fold CV)...")
    
    # 預處理：建立每季的 X, Y
    quarter_data = {}
    for q in unique_quarters:
        df_q = merged[merged['年季'] == q]
        # X: Index=Stock, Columns=Fund, Value=Weight
        X_matrix = df_q.pivot(index='key_code', columns='基金代碼', values='w').fillna(0)
        # Y: Stock Return
        stock_returns = df_q[['key_code', 'q_ret']].drop_duplicates().set_index('key_code')
        
        common = X_matrix.index.intersection(stock_returns.index)
        if len(common) < min_funds:
            quarter_data[q] = None
        else:
            quarter_data[q] = (X_matrix.loc[common], stock_returns.loc[common, 'q_ret'])

    final_results = []
    
    # 從第 lookback_quarters 季開始預測 (因為需要前 N 季當 Training)
    start_idx = lookback_quarters
    if start_idx >= len(unique_quarters):
        print("資料不足以進行滾動回測")
        return {}

    for i in range(start_idx, len(unique_quarters)):
        target_q = unique_quarters[i]   # 這是要預測的 Q(t)
        
        # 取得訓練視窗 (過去 N 季)
        train_qs = unique_quarters[i-lookback_quarters : i]
        
        # 1. 取得並合併訓練資料
        X_train_list, Y_train_list = [], []
        for tq in train_qs:
            d = quarter_data.get(tq)
            if d:
                X_train_list.append(d[0])
                Y_train_list.append(d[1])
        
        target_data = quarter_data.get(target_q)
        
        # 如果訓練資料為空或目標季資料缺失，則跳過
        if not X_train_list or target_data is None:
            continue
            
        X_train = pd.concat(X_train_list, sort=False).fillna(0)
        Y_train = pd.concat(Y_train_list)
        
        X_target, _ = target_data # 只需要 X
        
        # 2. 特徵對齊 (以 Training 為主)
        # 確保 Target 的 columns (基金) 跟 Training 一模一樣
        X_target = X_target.reindex(columns=X_train.columns, fill_value=0)
        
        # 3. 核心魔法：RidgeCV
        # cv=5: 自動將訓練資料隨機切成 5 份，做交叉驗證
        try:
            model = RidgeCV(alphas=ridge_alphas, cv=5, fit_intercept=True)
            model.fit(X_train, Y_train)
            
            # 4. 預測
            y_pred_target = model.predict(X_target)
            
            final_results.append(pd.DataFrame({
                '年季': target_q,
                'key_code': X_target.index,
                'Predicted_Score': y_pred_target,
                'Best_Alpha': model.alpha_ # 紀錄這一季選出的最佳 Alpha
            }))
        except Exception as e:
            print(f"Error at {target_q}: {e}")
            continue

    if not final_results: return {}
    
    combined_df = pd.concat(final_results, ignore_index=True)
    return {'Rolling_Best': combined_df}

# =========================
# 4. 回測工具 (Backtest)
# =========================
def prep_stock_monthly_for_backtest(df_stock, code_col='證券代碼', date_col='年月', ret_pct_col='報酬率％_月', price_col='收盤價(元)_月'):
    df = df_stock.copy()
    part = df[code_col].astype(str).str.strip().str.split(r'\s+', n=1, expand=True)
    df['標的碼'] = part[0]
    df['key_code'] = df['標的碼'].apply(extract_code_token)
    df['年月'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    df['ret_month'] = pd.to_numeric(df[ret_pct_col], errors='coerce') / 100
    df['price_month_end'] = pd.to_numeric(df.get(price_col, np.nan), errors='coerce')
    return df[['key_code', '年月', 'ret_month', 'price_month_end']].dropna(subset=['年月', 'key_code'])

def monthly_to_quarter_return(df_monthly):
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q')
    qret = d.groupby(['key_code', '年季'], as_index=False).agg(
        q_ret=('ret_month', lambda s: (1 + s).prod() - 1),
        n_months=('ret_month', 'count')
    )
    return qret[qret['n_months'] == 3].reset_index(drop=True)

def build_entry_eligibility(stock_m, min_price):
    sm = stock_m.copy()
    sm['年季'] = sm['年月'].dt.to_period('Q')
    sm['is_qe'] = sm['年月'].dt.is_quarter_end
    qe = sm[sm['is_qe']].copy()
    qe['持有季'] = (qe['年季'] + 1).astype('period[Q]')
    qe['eligible'] = (qe['price_month_end'] >= float(min_price)).astype(int)
    return qe[['key_code', '持有季', 'eligible']]

def _newey_west_t(series, lags=NW_LAGS):
    y = pd.Series(series).dropna()
    if len(y) < 5: return np.nan, np.nan, len(y)
    res = sm.OLS(y.values, np.ones((len(y), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0]), len(y)

def backtest_single_decile(gia_df, qret_df, eligibility_df, score_col, n_group=10, nw_lags=NW_LAGS):
    g = gia_df.copy()
    if not pd.api.types.is_period_dtype(g['年季']): g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq):
        dfq = dfq.copy()
        try:
            dfq['group'] = pd.qcut(dfq[score_col].rank(method='first'), q=n_group, labels=False, duplicates='drop') + 1
        except: dfq['group'] = np.nan
        return dfq

    g_grp = g.groupby('年季', group_keys=False).apply(assign_groups).reset_index(drop=True)
    g_grp = g_grp.rename(columns={'年季': 'formation_q'})
    g_grp['持有季'] = g_grp['formation_q'] + 1

    if eligibility_df is not None:
        elig = eligibility_df.copy()
        if not pd.api.types.is_period_dtype(elig['持有季']): elig['持有季'] = pd.PeriodIndex(elig['持有季'], freq='Q')
        g_grp = g_grp.merge(elig, on=['key_code', '持有季'], how='left')
        g_grp = g_grp[g_grp['eligible'].fillna(0).eq(1)]

    merged = pd.merge(g_grp, qret_df, left_on=['key_code', '持有季'], right_on=['key_code', '年季'], how='left')
    port = (merged.dropna(subset=['q_ret']).groupby(['formation_q', 'group'], as_index=False).agg(ret_mean=('q_ret', 'mean')))

    if port.empty: return pd.DataFrame(), pd.DataFrame()

    wide = port.pivot(index='formation_q', columns='group', values='ret_mean').sort_index()
    for k in range(1, n_group + 1):
        if k not in wide.columns: wide[k] = np.nan
    wide = wide[sorted(wide.columns)]
    wide['long_short'] = wide[n_group] - wide[1]

    rows = []
    for col in list(range(1, n_group + 1)) + ['long_short']:
        m, t, T = _newey_west_t(wide[col], lags=nw_lags)
        rows.append({'portfolio': col, 'mean': m, 't': t})
    summary = pd.DataFrame(rows).set_index('portfolio')
    return wide, summary

def calc_monotonicity_score(wide, n_group=10):
    cols = list(range(1, n_group + 1))
    mean_rets = wide[cols].mean()
    if mean_rets.isna().all(): return -1, 999
    rho = mean_rets.corr(pd.Series(range(1, n_group + 1), index=cols), method='spearman')
    rets_list = mean_rets.values
    violations = 0
    for i in range(len(rets_list)-1):
        if pd.isna(rets_list[i]) or pd.isna(rets_list[i+1]): continue
        if rets_list[i] > rets_list[i+1]: violations += 1
    return rho, violations

def build_slim_metrics_table(wide, summary_raw, periods_per_year=4):
    cols = [*(sorted([c for c in wide.columns if isinstance(c, int)])), 'long_short']
    out = []
    for col in cols:
        r = wide[col].dropna() if col in wide.columns else pd.Series(dtype=float)
        if r.empty:
            mean_q = std_q = mtv = sharpe_ann = np.nan
        else:
            mean_q = r.mean()
            std_q  = r.std(ddof=1)
            mtv = np.nan if (pd.isna(std_q) or std_q == 0) else mean_q / std_q
            sharpe_ann = np.nan if pd.isna(mtv) else mtv * np.sqrt(periods_per_year)
        tval = summary_raw.loc[col, 't'] if col in summary_raw.index else np.nan
        out.append([col, mean_q * 100, std_q * 100, mtv, sharpe_ann, tval])

    slim_num = pd.DataFrame(out, columns=['portfolio','mean_pct','std','mean_to_vol','sharpe_annual','t值']).set_index('portfolio')
    fmt_pct = lambda x: "" if pd.isna(x) else f"{x:.2f}%"
    fmt_val = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    fmt_t   = lambda x: "" if pd.isna(x) else f"{x:.2f}"
    slim_fmt = pd.DataFrame({
        'mean_pct':      slim_num['mean_pct'].apply(fmt_pct),
        'std':           slim_num['std'].apply(fmt_pct),
        'mean_to_vol':   slim_num['mean_to_vol'].apply(fmt_val),
        'sharpe_annual': slim_num['sharpe_annual'].apply(fmt_val),
        't值':           slim_num['t值'].apply(fmt_t),
    }, index=slim_num.index)
    return slim_num, slim_fmt

# =========================
# 5. 主程式
# =========================
def main():
    print(f"=== 啟動 ML RidgeCV (Cross-Sectional CV) ===")
    print(f"設定: Ridge Alpha Candidates={RIDGE_ALPHA_LIST}")
    
    try:
        print("讀取檔案...")
        df_fund = pd.read_csv("fund_data/merged_fund_data.csv", encoding='utf-8')
        df_holding = pd.read_csv("fund_data/fund_data.csv", encoding='utf-8')
        df_stock = pd.read_csv('fund_data/stock_return.csv', encoding='UTF-16 LE', sep='\t')
    except Exception as e:
        print(f"讀取錯誤 (請確認路徑): {e}")
        return

    # --- Preprocessing ---
    print("靜態資料前處理...")
    holding_data = prep_holding_from_fund_data(df_holding)
    stock_m = prep_stock_monthly_for_backtest(df_stock)
    stock_q = monthly_to_quarter_return(stock_m)
    entry_elig = build_entry_eligibility(stock_m, min_price=MIN_PRICE)
    
    if stock_q.empty:
        print("Stock Data 為空，無法執行。")
        return

    results = []
    cache_results = {} 
    total_start = time.time()
    
    # 1. 執行機器學習預測 (RidgeCV)
    # 不再需要自己迴圈，RidgeCV 會在內部幫你找最佳參數
    all_pred_results = run_ml_stock_prediction_ridge_cv(
        holding_w=holding_data,
        stock_q=stock_q,
        ridge_alphas=RIDGE_ALPHA_LIST,
        lookback_quarters = 2  # <--- 您可以在這裡修改要回看幾季
    )
    
    # 2. 回測
    print(f"   批次回測 Rolling Best Alpha 結果...")
    if 'Rolling_Best' in all_pred_results:
        pred_df = all_pred_results['Rolling_Best']
        
        # 使用 'Predicted_Score' 進行 Decile 排序
        wide, summary = backtest_single_decile(
            pred_df, 
            stock_q, 
            entry_elig, 
            score_col='Predicted_Score', 
            n_group=10, 
            nw_lags=NW_LAGS
        )
        
        if not wide.empty:
            rho, viol = calc_monotonicity_score(wide, n_group=10)
            ls_t = summary.loc['long_short', 't']
            
            results.append({
                'ridge_alpha': 'Rolling_CV_Adaptive', 
                'rho': rho,
                'viol': viol,
                't': ls_t
            })
            cache_results['Rolling_Best'] = (wide, summary)

    # --- Output ---
    if not results:
        print("無有效結果")
        return

    best_param = 'Rolling_Best'
    
    print("\n" + "="*60)
    print(f"【滾動驗證最佳結果 (RidgeCV)】")
    print(f"  > Strategy    : Cross-Sectional CV (1-Quarter Lookback)")
    print(f"  > Rho         : {results[0]['rho']:.4f}")
    print("="*60 + "\n")

    best_wide, best_summary = cache_results[best_param]
    _, slim_fmt = build_slim_metrics_table(best_wide, best_summary)
    
    print("\n" + "="*60)
    print(f"=== 最終績效報表 (Walk-Forward) ===")
    print("="*60)
    print(slim_fmt)
    print("="*60)
    
    print(f"\n全部完成，總耗時: {time.time() - total_start:.2f} 秒")

if __name__ == "__main__":
    main()