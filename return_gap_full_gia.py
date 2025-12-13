import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import warnings
import time

warnings.filterwarnings('ignore')

# =========================
# 0. 參數設定
# =========================
MIN_PRICE = 10      # 固定價格
NW_LAGS = 4         # Newey-West Lag
# K_RATIO 掃描範圍
K_RATIO_LIST = np.arange(0.01, 1, 0.01)

# =========================
# 1. 基礎工具函數
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

# =========================
# 2. 資料前處理
# =========================
def prepare_fund_data(df_fund, code_col='證券代碼', date_col='年月', ret_col='單月ROI', pct_as_percent=True):
    df = df_fund.copy()
    df[code_col] = df[code_col].astype(str).str.strip()
    df[date_col] = to_month_end(df[date_col])
    df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
    if pct_as_percent: df[ret_col] /= 100
    return df[[code_col, date_col, ret_col]].rename(columns={code_col:'基金代碼', ret_col:'ret_month'})

def prep_holding_from_fund_data(fund_data, fund_col='證券代碼', q_col='年季', stock_id='標的碼', weight_col='投資比率％',
                                keep_asset_col='投資標的', keep_asset_value='股票型'):
    h = fund_data.copy()
    if keep_asset_col in h.columns:
        h = h[h[keep_asset_col].astype(str).str.contains(str(keep_asset_value), na=False)]
    h['基金代碼'] = h[fund_col].astype(str).str.strip()
    h['年季'] = pd.PeriodIndex(h[q_col].astype(str).str.strip(), freq='Q')
    h['key_code'] = h[stock_id].astype(str).apply(extract_code_token)
    h['w_raw'] = pd.to_numeric(h[weight_col], errors='coerce') / 100
    h = h.groupby(['基金代碼','年季','key_code'], as_index=False).agg(w_raw=('w_raw','sum'))
    h = h[h['w_raw'] > 0].copy()
    h['w'] = h['w_raw'] / h.groupby(['基金代碼','年季'])['w_raw'].transform('sum')
    return h[['基金代碼','年季','key_code','w']]

def prep_stock_monthly_for_backtest(df_stock, code_col='證券代碼', date_col='年月',
                                    ret_pct_col='報酬率％_月', price_col='收盤價(元)_月', min_price=0):
    df = df_stock.copy()
    part = df[code_col].astype(str).str.strip().str.split(r'\s+', n=1, expand=True)
    df['key_code'] = part[0].apply(extract_code_token)
    df['年月'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    df['ret_month'] = pd.to_numeric(df[ret_pct_col], errors='coerce') / 100
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df[df[price_col] >= min_price]
    return df[['key_code','年月','ret_month']].dropna(subset=['ret_month'])

def monthly_to_quarter_return(df_monthly, key_col='key_code', ret_col='ret_month'):
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q')
    qret = d.groupby([key_col, '年季'], as_index=False).agg(
        q_ret=(ret_col, lambda s: (1+s).prod() - 1),
        n_months=(ret_col, 'count')
    )
    return qret[qret['n_months'] == 3].reset_index(drop=True)

# =========================
# 3. 核心邏輯 (Return Gap & GIA)
# =========================
def compute_return_gap(fund_monthly_returns, stock_quarterly_returns, fund_holdings):
    fund_q_ret = fund_monthly_returns.groupby(['基金代碼', pd.Grouper(key='年月', freq='Q')]).agg(
        fund_q_ret_actual=('ret_month', lambda s: (1+s).prod() - 1)
    ).reset_index()
    fund_q_ret['年季'] = fund_q_ret['年月'].dt.to_period('Q')

    holdings_begin_q = fund_holdings.copy()
    holdings_begin_q['年季'] = holdings_begin_q['年季'] + 1 
    merged = pd.merge(holdings_begin_q, stock_quarterly_returns, on=['key_code','年季'], how='inner')
    
    bh = merged.groupby(['基金代碼','年季']).apply(
        lambda g: np.average(g['q_ret'], weights=g['w'])
    ).reset_index(name='fund_q_ret_bh')

    gap = pd.merge(fund_q_ret, bh, on=['基金代碼','年季'], how='inner')
    gap['GAP'] = gap['fund_q_ret_actual'] - gap['fund_q_ret_bh']
    return gap[['基金代碼','年季','GAP']]

def compute_gia(alpha_q, holding_w, lag_holdings=True, k_ratio=0.5, min_funds=10):
    def _truncated_pinv(W, k):
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        k = max(1, min(int(np.floor(k)), len(s)))
        return (Vt[:k].T / s[:k]) @ U[:, :k].T

    A = alpha_q.copy()
    H = holding_w.copy()
    if lag_holdings: H['年季'] = (H['年季'] + 1).astype('period[Q]')
        
    out = []
    common_quarters = sorted(set(A['年季']) & set(H['年季']))
    for q in common_quarters:
        a_q = A.loc[A['年季'] == q, ['基金代碼', 'alpha']].dropna()
        h_q = H.loc[H['年季'] == q, ['基金代碼', 'key_code', 'w']]
        h_q = h_q[h_q['基金代碼'].isin(a_q['基金代碼'])].copy()
        
        funds = a_q['基金代碼'].unique(); M = len(funds)
        if M < min_funds: continue
        stocks = h_q['key_code'].unique(); N = len(stocks)
        if N == 0: continue
        
        f_id = {f: i for i, f in enumerate(funds)}
        s_id = {s: i for i, s in enumerate(stocks)}
        W = np.zeros((M, N))
        for _, r in h_q.iterrows(): W[f_id[r['基金代碼']], s_id[r['key_code']]] = r['w']
            
        S = a_q.set_index('基金代碼').reindex(funds)['alpha'].to_numpy(float)
        K = max(1, int(np.floor(k_ratio * M)))
        alpha_stock = _truncated_pinv(W, K) @ S
        out.append(pd.DataFrame({'年季': q, 'key_code': stocks, 'GIA': alpha_stock}))
        
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# =========================
# 4. 回測與報表工具
# =========================
def _newey_west_t(series, lags=6):
    y = pd.Series(series).dropna()
    if len(y) < 5: return np.nan, np.nan, len(y)
    res = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0]), len(y)

def backtest_single_decile(gia_df, qret_df, gia_col='GIA', n_group=10, nw_lags=6):
    g = gia_df.copy()
    if not pd.api.types.is_period_dtype(g['年季']): g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq):
        dfq = dfq.copy()
        dfq['group'] = pd.qcut(dfq[gia_col].rank(method='first'), q=n_group, labels=False, duplicates='drop') + 1
        return dfq

    g_grp = g.groupby('年季', group_keys=False).apply(assign_groups).reset_index(drop=True)
    g_grp = g_grp.rename(columns={'年季': 'formation_q'})
    g_grp['持有季'] = g_grp['formation_q'] + 1

    qret = qret_df.copy()
    if not pd.api.types.is_period_dtype(qret['年季']): qret['年季'] = pd.PeriodIndex(qret['年季'], freq='Q')

    merged = pd.merge(g_grp, qret, left_on=['key_code', '持有季'], right_on=['key_code', '年季'], how='left')
    
    port = (merged.dropna(subset=['q_ret'])
                  .groupby(['formation_q', 'group'], as_index=False)
                  .agg(ret_mean=('q_ret', 'mean')))

    wide = port.pivot(index='formation_q', columns='group', values='ret_mean').sort_index()
    for k in range(1, n_group + 1):
        if k not in wide.columns: wide[k] = np.nan
    wide = wide[sorted(wide.columns)]
    wide['long_short'] = wide[n_group] - wide[1]

    # 計算統計量
    rows = []
    for col in list(range(1, n_group + 1)) + ['long_short']:
        m, t, T = _newey_west_t(wide[col], lags=nw_lags)
        rows.append({'portfolio': col, 'mean': m, 't': t})
    summary = pd.DataFrame(rows).set_index('portfolio')
    
    return wide, summary

def calc_monotonicity_score(wide, n_group=10):
    # 計算 Spearman Rho
    cols = list(range(1, n_group + 1))
    mean_rets = wide[cols].mean()
    ranks = pd.Series(range(1, n_group + 1), index=cols)
    spearman_rho = mean_rets.corr(ranks, method='spearman')
    
    # 計算 Violations
    rets_list = mean_rets.values
    violations = 0
    for i in range(len(rets_list)-1):
        if rets_list[i] > rets_list[i+1]: 
            violations += 1
    return spearman_rho, violations

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
    print(f"=== 啟動最佳化回測 (Price={MIN_PRICE}) ===")
    
    # --- 1. 讀取 ---
    try:
        print("讀取檔案...")
        df_fund = pd.read_csv("fund_data/merged_fund_data.csv", encoding='utf-8') 
        df_holding = pd.read_csv("fund_data/fund_data.csv", encoding='utf-8')
        df_stock = pd.read_csv("fund_data/stock_return.csv", encoding='utf-16', sep='\t') 
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    # --- 2. 前處理 ---
    print("資料前處理...")
    fund_m = prepare_fund_data(df_fund)
    holding = prep_holding_from_fund_data(df_holding)
    stock_m = prep_stock_monthly_for_backtest(df_stock, min_price=MIN_PRICE)
    stock_q = monthly_to_quarter_return(stock_m)
    
    if stock_q.empty:
        print("錯誤: 股票資料為空")
        return

    # --- 3. Return Gap & Alpha ---
    print("計算 Return Gap...")
    gap = compute_return_gap(fund_m, stock_q, holding)
    if gap.empty:
        print("錯誤: GAP 計算失敗")
        return
    gap['alpha'] = -gap['GAP'] # 反向策略

    # --- 4. Grid Search 尋找最佳 K ---
    print(f"開始搜尋最佳參數 (K範圍: {K_RATIO_LIST[0]:.2f}~{K_RATIO_LIST[-1]:.2f})...")
    results = []
    
    for k in K_RATIO_LIST:
        gia = compute_gia(gap, holding, lag_holdings=True, k_ratio=k)
        if gia.empty: continue
        
        wide, summary = backtest_single_decile(gia, stock_q, n_group=10, nw_lags=NW_LAGS)
        rho, viol = calc_monotonicity_score(wide, n_group=10)
        ls_t = summary.loc['long_short', 't']
        
        results.append({'k': k, 'rho': rho, 'viol': viol, 't': ls_t})
        # print(f"K={k:.2f} -> Rho={rho:.2f}, Viol={viol}") # 想要看進度可以打開

    if not results:
        print("無有效結果")
        return

    # 排序：優先找 Rho 大 (降序) 且 Viol 小 (升序) 的
    df_res = pd.DataFrame(results).sort_values(
        by=['rho', 'viol', 't'], ascending=[False, True, False]
    )
    
    best_k = df_res.iloc[0]['k']
    best_rho = df_res.iloc[0]['rho']
    
    print("\n" + "="*50)
    print(f"【最佳參數確認】")
    print(f"  > Best K_RATIO : {best_k:.2f}")
    print(f"  > Spearman Rho : {best_rho:.4f}")
    print("="*50 + "\n")

    # --- 5. 用最佳參數重跑並列印詳細表 ---
    print(f"正在使用 K={best_k:.2f} 生成完整報表...")
    
    final_gia = compute_gia(gap, holding, lag_holdings=True, k_ratio=best_k)
    final_wide, final_summary = backtest_single_decile(final_gia, stock_q, n_group=10, nw_lags=NW_LAGS)
    
    _, slim_fmt = build_slim_metrics_table(final_wide, final_summary)
    
    print("\n" + "="*60)
    print(f"=== 最終績效報表 (K={best_k:.2f}, Price={MIN_PRICE}) ===")
    print("="*60)
    print(slim_fmt)
    print("="*60)

if __name__ == "__main__":
    main()



    
'''
=== Grid Search（含單調性評分）Top 10 ===

   k_ratio  min_price  violations  spearman_rho   slope_t  LS_mean_%  LS_t
0     0.08         35           1      0.733333  2.786288       1.95  2.11
1     0.12         20           2      0.587879  2.089273       1.53  2.82
2     0.20         20           2      0.454545  1.585218       2.10  3.28
3     0.20         35           2      0.333333  0.704435       1.67  2.06
4     0.20         30           3      0.648485  1.894203       1.58  2.74
5     0.08         15           3      0.575758  1.828588       1.60  2.16
6     0.08         30           3      0.466667  1.570409       1.01  1.50
7     0.08         40           3      0.418182  1.309303       1.17  1.35
8     0.10         10           3      0.406061  1.434928       1.29  1.92
9     0.12         15           3      0.406061  1.424944       1.26  2.19
'''
