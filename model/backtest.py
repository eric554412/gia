import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re
import os
from scipy.stats import spearmanr

# 解決 Matplotlib 中文顯示問題
import matplotlib
plt.rcParams['axes.unicode_minus'] = False
if os.name == 'nt': # Windows
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
else: # Mac / Linux
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']

# ==========================================
# 1. 全域參數設定
# ==========================================
MIN_PRICE = 10         
NW_LAGS = 4            
N_GROUPS = 10          

FACTOR_COLS = [
    'feat_TradingMoM',
    'feat_HBM',
    'feat_HoldingAlpha',
    'feat_ReturnGap',
]

# ==========================================
# 工具函數區塊
# ==========================================

def _newey_west_t(series, lags=6):
    '''計算 Newey-West t 統計量'''
    y = pd.Series(series).dropna()
    if len(y) < 5:
        return np.nan, np.nan, len(y)
    res = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0]), len(y)

def extract_code_token(s: str):
    s = str(s).strip().upper()
    if not s: return ""
    tok = s.split()[0].split('-')[0]
    return re.sub(r'[^A-Z0-9]', '', tok)

def calc_max_drawdown(return_series):
    '''計算最大回撤 (MDD)'''
    wealth_index = (1 + return_series).cumprod()
    peaks = wealth_index.cummax()
    drawdowns = (wealth_index - peaks) / peaks
    return drawdowns.min()

def prep_stock_monthly_for_backtest(df_stock: pd.DataFrame, code_col='證券代碼', date_col='年月',
                                    ret_pct_col='報酬率％_月', price_col='收盤價(元)_月'):
    df = df_stock.copy()
    part = df[code_col].astype(str).str.strip().str.split(r'\s+', n=1, expand=True)
    df['key_code'] = part[0].apply(extract_code_token)
    
    df['年月'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    
    df['ret_month'] = pd.to_numeric(df[ret_pct_col], errors='coerce') / 100
    df['price_month_end'] = pd.to_numeric(df.get(price_col, np.nan), errors='coerce')
    return df[['key_code', '年月', 'ret_month', 'price_month_end']].dropna(subset=['ret_month'])

def prep_benchmark_0050(file_path):
    '''讀取並處理 0050 ETF 資料'''
    try:
        df = pd.read_csv(file_path, encoding='utf-16', sep='\t') 
    except Exception:
        try:
             df = pd.read_csv(file_path, encoding='ms950')
        except Exception as e:
            print(f"讀取 0050 失敗: {e}")
            return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    
    ret_col = next((c for c in df.columns if '報酬' in c), None)
    date_col = next((c for c in df.columns if '年月' in c or '日期' in c), None)
    
    if not ret_col or not date_col:
        return pd.DataFrame()

    df['年月'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    df['ret_month'] = pd.to_numeric(df[ret_col], errors='coerce') / 100
    df = df.dropna(subset=['ret_month'])
    
    df['年季'] = df['年月'].dt.to_period('Q')
    
    qret = df.groupby('年季', as_index=False).agg(
        q_ret=('ret_month', lambda x: (1 + x).prod() - 1)
    )
    
    qret['key_code'] = '0050'
    return qret[['key_code', '年季', 'q_ret']]

# === [修改 1] 讀取並處理因子資料 (包含 RF) ===
def prep_factor_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-16', sep='\t')
    except Exception:
        try:
             df = pd.read_csv(file_path, encoding='utf-8') 
        except Exception as e:
            print(f"[警告] 讀取因子資料失敗: {e} (若無因子檔將跳過 Alpha 檢定)")
            return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    
    # 建立欄位對照表
    column_map = {
        '市場風險溢酬': 'MKT',
        '規模溢酬 (3因子)': 'SMB',
        '淨值市價比溢酬': 'HML',
        '動能因子': 'MOM',
        '年月': 'DATE'
    }
    
    # 自動搜尋無風險利率欄位 (TEJ 可能命名不同)
    rf_col_found = False
    for col in df.columns:
        if '無風險利率' in col:
            column_map[col] = 'RF'
            rf_col_found = True
            break
            
    df = df.rename(columns=column_map)
    
    # 若找不到 RF，建立全為 0 的欄位並警告
    if not rf_col_found and 'RF' not in df.columns:
        print("[注意] 因子檔中未找到'無風險利率'，假設 RF=0 (Alpha 可能被高估)")
        df['RF'] = 0.0

    req_factors = ['MKT', 'SMB', 'HML', 'MOM', 'RF']
    
    if 'DATE' not in df.columns:
        date_col = next((c for c in df.columns if '年月' in c or '日期' in c), None)
        if date_col:
            df = df.rename(columns={date_col: 'DATE'})
        else:
            print("[警告] 因子檔找不到日期欄位")
            return pd.DataFrame()
            
    # 處理日期
    df['年月'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m', errors='coerce')
    df['年季'] = df['年月'].dt.to_period('Q')
    
    # 處理數值 (除以100)
    for col in req_factors:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    def compound_ret(x):
        return (1 + x).prod() - 1
        
    # 將因子從月頻率轉為季頻率 (複利計算)
    q_factors = df.groupby('年季')[req_factors].agg(compound_ret).reset_index()
    return q_factors

# === [修改 2] 計算 Carhart Alpha (扣除 RF) ===
def calc_all_alphas(wide_df, df_factors):
    '''
    對 wide_df 中的每一個 column (分組) 計算 Carhart Alpha 和 Beta
    修正：針對非 Long-Short 投資組合，Y = Return - RF (超額報酬)
    '''
    results = {}
    
    if df_factors.empty:
        return results

    # 確保 Index 對齊
    if not isinstance(wide_df.index, pd.PeriodIndex):
        wide_df.index = pd.PeriodIndex(wide_df.index, freq='Q')
        
    df_factors = df_factors.set_index('年季')
    if not isinstance(df_factors.index, pd.PeriodIndex):
        df_factors.index = pd.PeriodIndex(df_factors.index, freq='Q')
        
    # 合併因子資料
    merged = wide_df.join(df_factors, how='inner').dropna()
    
    if len(merged) < 12: # 樣本太少不跑
        return results
        
    X = merged[['MKT', 'SMB', 'HML', 'MOM']]
    X = sm.add_constant(X) # 加上截距項 (Alpha)
    
    # 對每一欄位 (分組) 跑迴歸
    for col in wide_df.columns:
        if col not in merged.columns: continue
        
        # === 關鍵修正邏輯 ===
        # 1. 確保 RF 存在 (若 prep_factor_data 沒抓到會是 0)
        rf_correction = merged['RF'] if 'RF' in merged.columns else 0
        
        # 2. 定義 Y (被解釋變數)
        if str(col) == 'long_short':
            # 多空策略本身已中立掉 RF，不需要減
            Y = merged[col]
        else:
            # 單邊策略 (Decile 1-10, ETF) 需減 RF 轉為超額報酬
            Y = merged[col] - rf_correction
        
        try:
            # 使用 Newey-West 調整標準誤
            model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            
            alpha = model.params['const']
            t_stat = model.tvalues['const']
            beta = model.params['MKT'] # 抓取 Beta
            
            results[col] = (alpha, t_stat, beta)
        except Exception:
            results[col] = (np.nan, np.nan, np.nan)
            
    return results

def build_entry_eligibility(stock_m: pd.DataFrame, min_price):
    sm = stock_m.copy()
    sm['年季'] = sm['年月'].dt.to_period('Q')
    sm['is_qe'] = sm['年月'].dt.is_quarter_end
    qe = sm[sm['is_qe']].copy()
    qe['持有季'] = (qe['年季'] + 1).astype('period[Q]')
    qe['eligible'] = (qe['price_month_end'] >= min_price).astype(int)
    return qe[['key_code', '持有季', 'eligible']]    

def monthly_to_quarterly_return(df_monthly: pd.DataFrame, key_col='key_code', ret_col='ret_month'):
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q')
    qret = d.groupby([key_col, '年季'], as_index=False).agg(
        q_ret=(ret_col, lambda x: (1 + x).prod() - 1),
        n_months=(ret_col, 'count')
    )
    return qret[qret['n_months'] == 3].reset_index(drop=True)

def backtest_composite_score(score_df, qret_df, eligibility_df=None, score_col='composite_score', n_groups=5, nw_lags=NW_LAGS):
    g = score_df.copy()
    if not isinstance(g['年季'].dtype, pd.PeriodDtype):
         g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq):
        try:
            dfq['group'] = pd.qcut(
                dfq[score_col].rank(method='first'),
                q=n_groups,
                labels=False,
                duplicates='drop'
            ) + 1
        except Exception:
             dfq['group'] = np.nan
        return dfq

    g_grp = g.groupby('年季', group_keys=False).apply(assign_groups).reset_index(drop=True)
    g_grp = g_grp.rename(columns={'年季': 'formation_q'})
    g_grp['持有季'] = g_grp['formation_q'] + 1
    
    if eligibility_df is not None:
        elig = eligibility_df.copy()
        if not isinstance(elig['持有季'].dtype, pd.PeriodDtype):
             elig['持有季'] = pd.PeriodIndex(elig['持有季'], freq='Q')
        g_grp = g_grp.merge(elig, on=['key_code', '持有季'], how='left')
        g_grp = g_grp[g_grp['eligible'].fillna(0) == 1]
    
    qret = qret_df.copy()
    if not isinstance(qret['年季'].dtype, pd.PeriodDtype):
        qret['年季'] = pd.PeriodIndex(qret['年季'], freq='Q')

    merged = pd.merge(g_grp, qret, left_on=['key_code', '持有季'], right_on=['key_code', '年季'], how='left')
    valid_merged = merged.dropna(subset=['q_ret'])

    avg_total_stocks = valid_merged.groupby('formation_q')['key_code'].nunique().mean()

    port = (valid_merged
                  .groupby(['formation_q', 'group'], as_index=False)
                  .agg(ret_mean=('q_ret', 'mean'), n_stocks=('key_code', 'count')))

    wide = port.pivot(index='formation_q', columns='group', values='ret_mean').sort_index()
    avg_stocks_per_group = port.groupby('group')['n_stocks'].mean()
    
    stock_sets = valid_merged.groupby(['formation_q', 'group'])['key_code'].apply(set).unstack('group')
    turnover_df = pd.DataFrame(index=stock_sets.index, columns=stock_sets.columns, dtype=float)
    
    for group in stock_sets.columns:
        prev_set = set()
        for q in stock_sets.index:
            curr_set = stock_sets.at[q, group]
            if isinstance(curr_set, set) and len(curr_set) > 0:
                if len(prev_set) > 0:
                    new_stocks = curr_set - prev_set
                    turnover_df.at[q, group] = len(new_stocks) / len(curr_set)
            prev_set = curr_set if isinstance(curr_set, set) else set()
            
    avg_turnover_per_group = turnover_df.mean()

    for k in range(1, n_groups + 1):
        if k not in wide.columns: wide[k] = np.nan
    wide = wide[sorted(wide.columns)]
    
    if n_groups in wide.columns and 1 in wide.columns:
        wide['long_short'] = wide[n_groups] - wide[1]
    else:
        wide['long_short'] = np.nan

    rows = []
    cols = list(range(1, n_groups + 1)) + ['long_short']
    for col in cols:
        if col in wide.columns:
            m, t, _ = _newey_west_t(wide[col], lags=nw_lags)
            avg_n = avg_stocks_per_group.get(col, np.nan) if isinstance(col, int) else np.nan
            turnover = avg_turnover_per_group.get(col, np.nan) if isinstance(col, int) else np.nan
            
            rows.append({
                'portfolio': col, 
                'mean': m, 
                't': t, 
                'avg_stocks': avg_n,
                'turnover': turnover
            })
            
    summary = pd.DataFrame(rows).set_index('portfolio')
    return wide, summary, avg_total_stocks

def build_slim_metrics_table(wide, summary_raw, alpha_results, periods_per_years=4):
    cols = [*sorted(c for c in wide.columns if isinstance(c, int)), 'long_short']
    if '0050 ETF' in wide.columns:
        cols.append('0050 ETF')
        
    out = []
    for col in cols:
        r = wide[col] if col in wide.columns else pd.Series(dtype=float)
        r_clean = r.dropna()
        
        if r_clean.empty:
            mean_q = std_q = mtv = sharpe_ann = total_ret = mdd = vol_ann = np.nan
        else:
            mean_q = r_clean.mean()
            std_q = r_clean.std()
            vol_ann = std_q * np.sqrt(periods_per_years)
            mtv = np.nan if (pd.isna(std_q) or std_q == 0) else mean_q / std_q 
            sharpe_ann = np.nan if (pd.isna(std_q) or std_q == 0) else mtv * np.sqrt(periods_per_years)
            total_ret = (1 + r_clean).prod() - 1 
            mdd = calc_max_drawdown(r_clean)      
            
        if col == '0050 ETF':
            tval = avg_stocks = turnover = np.nan 
        else:
            tval = summary_raw.loc[col, 't'] if col in summary_raw.index else np.nan
            avg_stocks = summary_raw.loc[col, 'avg_stocks'] if col in summary_raw.index else np.nan
            turnover = summary_raw.loc[col, 'turnover'] if col in summary_raw.index else np.nan

        # alpha_results 格式: {col: (alpha, t_stat, beta)}
        alpha_val, alpha_t, beta_val = alpha_results.get(col, (np.nan, np.nan, np.nan))
        
        out.append([col, mean_q * 100, std_q * 100, vol_ann * 100, sharpe_ann, tval, total_ret * 100, mdd * 100, avg_stocks, turnover, alpha_val * 100, alpha_t, beta_val])
    
    columns = ['portfolio', 'mean_pct', 'std_pct', 'vol_ann_pct', 'sharpe_annual', 't值', 'total_ret_pct', 'mdd_pct', 'avg_stocks', 'turnover', 'alpha_pct', 'alpha_t', 'beta']
    slim_num = pd.DataFrame(out, columns=columns).set_index('portfolio')
    
    fmt_pct = lambda x: "" if pd.isna(x) else f"{x:.2f}%"
    fmt_pct_turnover = lambda x: "" if pd.isna(x) else f"{x*100:.1f}%"
    fmt_val = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    fmt_t = lambda x: "" if pd.isna(x) else f"{x:.2f}"
    fmt_count = lambda x: "" if pd.isna(x) else f"{x:.0f}" 
    
    slim_fmt = pd.DataFrame({
        '季均檔數': slim_num['avg_stocks'].apply(fmt_count), 
        '季均換股率': slim_num['turnover'].apply(fmt_pct_turnover),
        '年化報酬': (slim_num['mean_pct'] * periods_per_years / 100).apply(lambda x: f"{x:.2%}" if not pd.isna(x) else ""),
        '季平均報酬': slim_num['mean_pct'].apply(fmt_pct), 
        '年化波動率': slim_num['vol_ann_pct'].apply(fmt_pct),
        '夏普比率': slim_num['sharpe_annual'].apply(fmt_val),
        't值': slim_num['t值'].apply(fmt_t),
        '總累積報酬': slim_num['total_ret_pct'].apply(fmt_pct),
        '最大回撤': slim_num['mdd_pct'].apply(fmt_pct),
        "Jensen's Alpha": slim_num['alpha_pct'].apply(fmt_pct),
        'Alpha t值': slim_num['alpha_t'].apply(fmt_t),
        'Market Beta': slim_num['beta'].apply(fmt_val) 
    })
    return slim_fmt

def calc_spearman_monotonicity(wide_df, n_groups):
    cols = list(range(1, n_groups + 1))
    valid_cols = [c for c in cols if c in wide_df.columns]
    if len(valid_cols) < 3: return np.nan, np.nan
    mean_rets = wide_df[valid_cols].mean()
    ranks = np.array(valid_cols)
    rho, p_value = spearmanr(ranks, mean_rets.values)
    return rho, p_value

def plot_performance(wide, n_groups):
    if wide.empty: return
    cum_ret = (1 + wide).cumprod()
    if not isinstance(cum_ret.index, pd.DatetimeIndex):
        cum_ret.index = cum_ret.index.to_timestamp()
    
    plt.figure(figsize=(12, 7))
    if 'long_short' in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret['long_short'], label='Long-Short', color='#d62728', linewidth=3.0, zorder=10)
    if '0050 ETF' in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret['0050 ETF'], label='Benchmark (0050 ETF)', color='black', linewidth=2.0, alpha=0.8)
    if n_groups in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret[n_groups], label=f'Top Decile (Group {n_groups})', color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    if 1 in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret[1], label='Bottom Decile (Group 1)', color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    plt.title('GIA因子策略績效', fontsize=16, fontweight='bold')
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('累積淨值', fontsize=12)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程式區塊
# ==========================================
def run_backtest(df_scores_input):
    print("===== 開始自選因子簡單平均回測 =====")
    
    try:
        df_stock = pd.read_csv("fund_data/stock_return.csv", encoding='utf-16', sep='\t')
        print("成功讀取股票報酬資料")
        
        df_0050_q = prep_benchmark_0050("fund_data/0050.csv")
        
        # 讀取因子資料
        df_factors = prep_factor_data("fund_data/carhart_factor.csv")
        if not df_factors.empty:
            print("成功讀取並處理 Carhart 4因子資料 (含 RF)")
            
    except Exception as e:
        print(f"錯誤：無法讀取資料 - {e}")
        return

    print("資料前處理...")
    stock_m = prep_stock_monthly_for_backtest(df_stock)
    stock_q = monthly_to_quarterly_return(stock_m)
    entry_elig = build_entry_eligibility(stock_m, MIN_PRICE)

    df_scores = df_scores_input.copy()
    df_scores['key_code'] = df_scores['key_code'].astype(str).str.strip()
    
    valid_factor_cols = [c for c in FACTOR_COLS if c in df_scores.columns]
    if not valid_factor_cols:
        print("錯誤: 沒有有效的因子欄位，停止回測。")
        return

    df_scores['composite_score'] = df_scores[valid_factor_cols].mean(axis=1)

    print("執行分組回測...")
    wide, summary, avg_total_stocks = backtest_composite_score(
        df_scores, 
        stock_q, 
        entry_elig, 
        score_col='composite_score', 
        n_groups=N_GROUPS,
        nw_lags=NW_LAGS
    )

    if not df_0050_q.empty:
        bench_series = df_0050_q.set_index('年季')['q_ret']
        if not isinstance(wide.index, pd.PeriodIndex):
             wide.index = pd.PeriodIndex(wide.index, freq='Q')
        wide['0050 ETF'] = bench_series.reindex(wide.index)

    # 計算 Alpha (已修正 RF)
    alpha_results = calc_all_alphas(wide, df_factors)

    slim_fmt = build_slim_metrics_table(wide, summary, alpha_results)
    rho, p_val = calc_spearman_monotonicity(wide, N_GROUPS)

    print("\n" + "="*90)
    print(f"=== 自選因子策略績效 (vs 0050 ETF) ===")
    print(f"=== [市場概況] 平均每季符合條件總檔數: {avg_total_stocks:.0f} 檔 ===")
    print(f"=== [單調性檢定] Spearman Rho: {rho:.4f} (p-value: {p_val:.4f}) ===")
    print("="*90)
    print(slim_fmt)
    print("="*90)
    
    print("正在繪製累積報酬圖...")
    plot_performance(wide, N_GROUPS)
    
    return wide

if __name__ == "__main__":
    # 請確保檔案路徑正確
    if os.path.exists("fund_data/gia_score.csv"):
        df_my_scores = pd.read_csv("fund_data/gia_score.csv") 
        if 'feat_ReturnGap' in df_my_scores.columns:
            df_my_scores['feat_ReturnGap'] = -df_my_scores['feat_ReturnGap']
            
        run_backtest(df_my_scores)
    else:
        print("找不到 gia_score.csv，請確認路徑。")