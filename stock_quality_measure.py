import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import warnings
import time

warnings.filterwarnings('ignore')

# =========================
# CONFIG (參數設定)
# =========================
WINDOW_LIST = [18]                         # Carhart Alpha 滾動窗口
K_RATIO_LIST = np.arange(0.05, 0.51, 0.01) # K Ratio 掃描範圍
MIN_PRICE = 10                             # 最低股價
NW_LAGS = 4                                # Newey-West Lags
LAG_HOLDINGS = 0                           # 模擬 1 季公告延遲

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
# 2. 資料前處理 (Carhart 四因子)
# =========================
def prepare_fund_data(df_fund, code_col='證券代碼', date_col='年月', ret_col='單月ROI', pct_as_percent=True):
    df = df_fund.copy()
    df[code_col] = df[code_col].astype(str).str.strip()
    df[date_col] = to_month_end(df[date_col])
    df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
    if pct_as_percent: df[ret_col] /= 100
    return df[[code_col, date_col, ret_col]]

def prepare_factor_data(df_factor, date_col='年月',
                        mktrf_col='市場風險溢酬', smb_col='規模溢酬 (3因子)',
                        hml_col='淨值市價比溢酬', mom_col='動能因子',
                        rf_col='無風險利率', pct_as_percent=True):
    fac = df_factor.copy()
    fac[date_col] = to_month_end(fac[date_col])
    
    # 讀取四因子
    cols_to_read = [mktrf_col, smb_col, hml_col, mom_col, rf_col]
    for c in cols_to_read:
        fac[c] = pd.to_numeric(fac[c], errors='coerce')
    
    if pct_as_percent:
        for c in [mktrf_col, smb_col, hml_col, mom_col]:
            fac[c] /= 100
            
    rf_annual = fac[rf_col] / 100 if pct_as_percent else fac[rf_col]
    fac['RF'] = rf_annual / 12
    
    # 重新命名
    fac = fac[[mktrf_col, smb_col, hml_col, mom_col, 'RF', date_col]].rename(
        columns={mktrf_col: 'MKT', smb_col: 'SMB', hml_col: 'HML', mom_col: 'MOM'}
    )
    return fac.sort_values(by=[date_col]).drop_duplicates(subset=[date_col], keep='last').reset_index(drop=True)

def merge_fund_factor(fund_df, fac_df, date_col='年月', code_col='證券代碼', ret_col='單月ROI'):
    merged = pd.merge(fund_df, fac_df, on=date_col, how='inner').sort_values([code_col, date_col]).reset_index(drop=True)
    merged['ret_excess'] = merged[ret_col] - merged['RF']
    return merged

# =========================
# 3. 核心運算: Carhart Alpha & GIA
# =========================
def run_carhart_rolling(merged_df, window=18, min_periods=None):
    if min_periods is None: min_periods = window
    df = merged_df.copy().sort_values(by=['證券代碼', '年月']).reset_index(drop=True)
    
    # === 使用四因子 ===
    x_cols = ['MKT', 'SMB', 'HML', 'MOM']
    
    out_frames = []
    for fid, g in df.groupby('證券代碼', sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < min_periods: continue
        
        y_all = g['ret_excess'].to_numpy(float)
        X_all = g[x_cols].to_numpy(float)
        alpha = np.full(n, np.nan)
        nobs = np.zeros(n, int)
        qe_mask = g['年月'].dt.is_quarter_end.to_numpy()
        
        for t in np.flatnonzero(qe_mask):
            end = t + 1
            start = max(0, end - window)
            if end - start < min_periods: continue
            y_win = y_all[start:end]
            X_win = X_all[start:end, :] 
            mask = np.isfinite(y_win) & np.all(np.isfinite(X_win), axis=1)
            if mask.sum() < min_periods: continue
            try:
                X_use = sm.add_constant(pd.DataFrame(X_win[mask], columns=x_cols), has_constant='add')
                res = sm.OLS(y_win[mask], X_use).fit()
                alpha[t] = res.params.get('const', np.nan)
                nobs[t] = int(res.nobs)
            except: pass
            
        sel = qe_mask
        out = pd.DataFrame({
            '年月': g.loc[sel, '年月'].values,
            '基金代碼': g.loc[sel, '證券代碼'].values,
            'alpha': alpha[sel],
            'n_obs': nobs[sel]
        })
        out['valid_alpha'] = (out['n_obs'] >= min_periods).astype(int)
        out_frames.append(out)
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

def clean_alpha_panel(df, valid_col='valid_alpha', alpha_col='alpha'):
    if df.empty: return pd.DataFrame()
    out = df[df[valid_col].eq(1)].copy()
    out[alpha_col] = pd.to_numeric(out[alpha_col], errors='coerce')
    return out.dropna(subset=[alpha_col])

def prepare_fund_alpha(alpha_df):
    if alpha_df.empty: return pd.DataFrame()
    a = alpha_df.copy()
    a['年季'] = to_quarter_end(a['年月']).dt.to_period('Q')
    return a.rename(columns={'基金代碼': '基金代碼', '證券代碼': '基金代碼'})[['基金代碼', '年季', 'alpha']]

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

def compute_gia_grid_optimized(alpha_q, holding_w, k_ratio_list, lag_holidings=0, min_funds=10):
    A = alpha_q.copy()
    H = holding_w.copy()
    if lag_holidings == 1: H['年季'] = (H['年季'] + 1).astype('period[Q]')
    
    results_dict = {k: [] for k in k_ratio_list}
    common_quarters = sorted(set(A['年季']) & set(H['年季']))
    print(f"      > GIA 計算中 (共 {len(common_quarters)} 季)...")
    
    for q in common_quarters:
        a_q = A.loc[A['年季'].eq(q), ['基金代碼', 'alpha']].dropna()
        h_q = H.loc[H['年季'].eq(q), ['基金代碼', 'key_code', 'w']]
        h_q = h_q[h_q['基金代碼'].isin(a_q['基金代碼'])].copy()
        
        funds = a_q['基金代碼'].unique(); M = len(funds)
        if M < min_funds: continue
        stocks = h_q['key_code'].unique(); N = len(stocks)
        if N == 0: continue
        
        f_id = {f: i for i, f in enumerate(funds)}
        s_id = {s: i for i, s in enumerate(stocks)}
        W = np.zeros((M, N))
        for _, r in h_q.iterrows(): W[f_id[r['基金代碼']], s_id[r['key_code']]] = r['w']
        S_vec = a_q.set_index('基金代碼').reindex(funds)['alpha'].to_numpy(float)
        
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        k_max_theoretical = int((s > 1e-10).sum()) 
        
        for k_ratio in k_ratio_list:
            K = max(1, int(np.floor(k_ratio * M)))
            K = min(K, k_max_theoretical) 
            right_part = U[:, :K].T @ S_vec 
            alpha_stock = (Vt[:K].T / s[:K]) @ right_part
            results_dict[k_ratio].append(pd.DataFrame({'年季': q, 'key_code': stocks, 'GIA': alpha_stock}))
            
    final_output = {}
    for k, frames in results_dict.items():
        if frames: final_output[k] = pd.concat(frames, ignore_index=True)
        else: final_output[k] = pd.DataFrame()
    return final_output

# =========================
# 4. 回測與報表工具 (使用 Return Gap 版本的格式)
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
    try:
        res = sm.OLS(y.values, np.ones((len(y), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        return float(res.params[0]), float(res.tvalues[0]), len(y)
    except:
        return np.mean(y), np.nan, len(y)

def backtest_single_decile(gia_df, qret_df, eligibility_df, n_group=10, nw_lags=NW_LAGS):
    g = gia_df.copy()
    if not pd.api.types.is_period_dtype(g['年季']): g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq):
        dfq = dfq.copy()
        try:
            dfq['group'] = pd.qcut(dfq['GIA'].rank(method='first'), q=n_group, labels=False, duplicates='drop') + 1
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
    if wide.empty: return -1, 999
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

# === 這是跟 Return Gap 版本完全一致的報表產生函數 ===
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
    print(f"=== 啟動 Full GIA (Carhart Version) Grid Search (Price={MIN_PRICE}) ===")
    print(f"設定: Window List={WINDOW_LIST}")
    print(f"設定: K Ratio Range={K_RATIO_LIST[0]:.2f}~{K_RATIO_LIST[-1]:.2f}")
    
    try:
        print("讀取檔案...")
        df_fund = pd.read_csv("fund_data/merged_fund_data.csv", encoding='utf-8')
        df_factor = pd.read_csv("fund_data/carhart_factor.csv", encoding='UTF-16 LE', sep='\t')
        df_holding = pd.read_csv("fund_data/fund_data.csv", encoding='utf-8')
        df_stock = pd.read_csv('fund_data/stock_return.csv', encoding='UTF-16 LE', sep='\t')
    except Exception as e:
        print(f"讀取錯誤 (請確認路徑): {e}")
        return

    print("靜態資料前處理...")
    fund_data = prepare_fund_data(df_fund)
    
    # 準備 Carhart 四因子
    factor_data = prepare_factor_data(df_factor) 
    merged_for_alpha = merge_fund_factor(fund_data, factor_data)
    
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
    
    for win in WINDOW_LIST:
        print(f"\n>>> 正在處理 Window = {win} ...")
        t0 = time.time()
        
        # 計算 Carhart Alpha
        print(f"   計算 Carhart Alpha (Window={win})...")
        coef = run_carhart_rolling(merged_for_alpha, window=win, min_periods=win)
        
        coef_clean = clean_alpha_panel(coef)
        alpha_df = prepare_fund_alpha(coef_clean)
        
        if alpha_df.empty:
            print(f"   [Warning] Window={win} 產生的 Alpha 為空，跳過。")
            continue
            
        # 計算 GIA
        all_gia_results = compute_gia_grid_optimized(
            alpha_q=alpha_df, 
            holding_w=holding_data, 
            k_ratio_list=K_RATIO_LIST,
            lag_holidings=LAG_HOLDINGS
        )
        
        # 回測
        print(f"   批次回測所有 K Ratios...")
        for k, gia_df in all_gia_results.items():
            if gia_df.empty: continue
            
            wide, summary = backtest_single_decile(gia_df, stock_q, entry_elig, n_group=10, nw_lags=NW_LAGS)
            if wide.empty: continue
            
            rho, viol = calc_monotonicity_score(wide, n_group=10)
            ls_t = summary.loc['long_short', 't']
            
            res_key = (win, k)
            results.append({'window': win, 'k_ratio': k, 'rho': rho, 'viol': viol, 't': ls_t})
            cache_results[res_key] = (wide, summary)
            
        print(f"   Window {win} 完成 (耗時 {time.time()-t0:.1f} 秒)")

    if not results:
        print("無有效結果")
        return

    df_res = pd.DataFrame(results).sort_values(by=['rho', 'viol', 't'], ascending=[False, True, False])
    
    best_row = df_res.iloc[0]
    best_win = int(best_row['window'])
    best_k   = float(best_row['k_ratio'])
    best_rho = best_row['rho']
    
    print("\n" + "="*60)
    print("=== Grid Search 結果排序 (Top 10) ===")
    print(df_res.head(10).to_string(index=False, float_format="{:.4f}".format))
    print("-" * 60)
    print(f"【最佳參數組合】")
    print(f"  > Window  : {best_win}")
    print(f"  > K Ratio : {best_k:.2f}")
    print(f"  > Rho     : {best_rho:.4f}")
    print("="*60 + "\n")

    # === 使用跟 Return Gap 一樣的報表格式函數 ===
    best_key = (best_win, best_k)
    best_wide, best_summary = cache_results[best_key]
    _, slim_fmt = build_slim_metrics_table(best_wide, best_summary)
    
    print("\n" + "="*60)
    print(f"=== 最終績效報表 (Window={best_win}, K={best_k:.2f}) ===")
    print("="*60)
    print(slim_fmt)
    print("="*60)
    print(f"\n全部完成，總耗時: {time.time() - total_start:.2f} 秒")

if __name__ == "__main__":
    main()
    

'''
 trim_p  min_price  violations  spearman_rho   slope_t  LS_mean_%      LS_t
0      0.0          0           2      0.600000  1.333956   1.036770  1.621949
1      0.0          5           2      0.539394  1.225296   1.022388  1.610571
2      0.0         10           2      0.369697  1.057400   1.058691  1.508262
3      0.0         30           3      0.660606  2.974742   1.432121  1.396900
4      0.0         20           3      0.466667  1.608909   1.029513  1.339443
5      0.0         15           3      0.321212  1.284729   1.079216  1.471605
6      0.0         40           3      0.248485  0.968322   0.811855  0.910526
7      0.0         35           4      0.551515  1.924425   0.955919  0.910709
8      0.0         25           4      0.515152  2.138689   1.799173  2.018110
9      0.0         50           4      0.369697  1.060742   0.836040  0.690707
10     0.0         45           4      0.224242  0.930678   1.048315  0.988810
'''