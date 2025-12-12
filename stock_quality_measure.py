import pandas as pd
import numpy as np
import statsmodels.api as sm
import re

# =========================
# 固定參數（單次跑）
# =========================
TRIM_P = 0.00      # 月度回報 cross-section trimming 比例（0=不剔除）
MIN_PRICE = 10     # 持有期一開始（持有季第一個月月末價）價格門檻
NW_LAGS = 4        # Newey-West 最大滯後

# -----------------------------
# 日期與資料前處理
# -----------------------------
def to_month_end(s):
    s_str = s.astype(str)
    d1 = pd.to_datetime(s_str, format='%Y%m', errors='coerce')
    d2 = pd.to_datetime(s_str, format='%Y-%m', errors='coerce')
    d3 = pd.to_datetime(s_str, errors='coerce')
    d = d1.fillna(d2).fillna(d3)
    return d.dt.to_period('M').dt.to_timestamp('M')

def prepare_fund_data(df_fund: pd.DataFrame, code_col='證券代碼', date_col='年月', ret_col='單月ROI', pct_as_percent=True):
    df = df_fund.copy()
    df[code_col] = df[code_col].astype(str).str.strip()
    df[date_col] = to_month_end(df[date_col])
    df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
    if pct_as_percent:
        df[ret_col] = df[ret_col] / 100
    return df[[code_col, date_col, ret_col]]

def prepare_factor_data(df_factor: pd.DataFrame, date_col='年月', 
                        mktrf_col='市場風險溢酬', rf_col='無風險利率', pct_as_percent=True):
    # CAPM 只需要 市場溢酬 (MKT) 和 無風險利率 (RF)
    fac = df_factor.copy()
    fac[date_col] = to_month_end(fac[date_col])
    
    # 這裡只讀取需要的欄位
    for c in [mktrf_col, rf_col]:
        fac[c] = pd.to_numeric(fac[c], errors='coerce')
    
    if pct_as_percent:
        fac[mktrf_col] = fac[mktrf_col] / 100
        
    rf_annual = fac[rf_col] / 100 if pct_as_percent else fac[rf_col]
    fac['RF'] = rf_annual / 12
    
    # 重新命名 MKT
    fac = fac[[mktrf_col, 'RF', date_col]].rename(columns={mktrf_col: 'MKT'})
    fac = fac.sort_values(by=[date_col]).drop_duplicates(subset=[date_col], keep='last').reset_index(drop=True)
    return fac

def merge_fund_factor(fund_df, fac_df, date_col='年月', code_col='證券代碼', ret_col='單月ROI'):
    merged = pd.merge(fund_df, fac_df, on=date_col, how='inner').sort_values([code_col, date_col]).reset_index(drop=True)
    merged['ret_excess'] = merged[ret_col] - merged['RF']
    return merged

# -----------------------------
# CAPM 單因子滾動回歸（基金 α）- 修改處
# -----------------------------
def run_capm_rolling(
    merged_df: pd.DataFrame, date_col='年月', code_col='證券代碼',
    y_col='ret_excess', x_cols=('MKT',),  # 這裡只保留 MKT
    window=18, min_periods=18, cov_type=None):
    
    df = merged_df.copy().sort_values(by=[code_col, date_col]).reset_index(drop=True)
    out_frames = []
    
    # 確保 x_cols 是 list
    x_cols = list(x_cols)
    
    for fid, g in df.groupby(code_col, sort=False):
        g = g.reset_index(drop=True)
        y_all = g[y_col].to_numpy(float)
        X_all = g[x_cols].to_numpy(float)
        n = len(g)
        
        alpha = np.full(n, np.nan)
        betas = np.full((n, len(x_cols)), np.nan)
        r2s = np.full(n, np.nan)
        nobs = np.zeros(n, int)
        qe_mask = g[date_col].dt.is_quarter_end.to_numpy()
        
        for t in np.flatnonzero(qe_mask):
            end = t + 1
            start = max(0, end - window)
            if end - start < min_periods:
                continue
            
            y_win = y_all[start:end]
            X_win = X_all[start:end, :]
            
            mask = np.isfinite(y_win) & np.all(np.isfinite(X_win), axis=1)
            if mask.sum() < min_periods:
                continue
            
            # CAPM 回歸: Excess Ret ~ const + MKT
            X_use = sm.add_constant(pd.DataFrame(X_win[mask], columns=x_cols), has_constant='add')
            
            if cov_type:
                res = sm.OLS(y_win[mask], X_use, missing='drop').fit(cov_type=cov_type)
            else:
                res = sm.OLS(y_win[mask], X_use).fit()
            
            alpha[t] = res.params.get('const', np.nan)
            
            # 儲存 Beta (CAPM 只有 Beta_MKT)
            for i, c in enumerate(x_cols):
                betas[t, i] = res.params.get(c, np.nan)
                
            r2s[t] = res.rsquared
            nobs[t] = int(res.nobs)
            
        sel = qe_mask
        out = pd.DataFrame({
            date_col: g.loc[sel, date_col].to_numpy(),
            code_col: g.loc[sel, code_col].to_numpy(),
            'alpha': alpha[sel],
            **{f'beta_{c.lower()}': betas[sel, i] for i, c in enumerate(x_cols)},
            'r2': r2s[sel],
            'n_obs': nobs[sel]
        })
        out['valid_alpha'] = (out['n_obs'] >= min_periods).astype(int)
        out_frames.append(out)
        
    return pd.concat(out_frames, ignore_index=True)

def clean_alpha_panel(df: pd.DataFrame):
    out = df[df['valid_alpha'].eq(1)].copy()
    beta_cols = [c for c in df.columns if c.startswith('beta_')]
    cols = ['證券代碼', '年月', 'alpha'] + beta_cols + ['r2', 'n_obs', 'valid_alpha']
    out = out[cols].dropna(subset=['alpha']).sort_values(['證券代碼', '年月']).reset_index(drop=True)
    return out

# -----------------------------
# Fund α、持股處理
# -----------------------------
def to_quarter_end(s):
    s = pd.Series(s)
    dt = pd.to_datetime(s, errors='coerce')
    return dt.dt.to_period('Q').dt.to_timestamp('Q')

def prepare_fund_alpha(alpha_df: pd.DataFrame):
    a = alpha_df[alpha_df['valid_alpha'].eq(1)].copy()
    a['年季'] = to_quarter_end(a['年月']).dt.to_period('Q')
    a['alpha'] = pd.to_numeric(a['alpha'], errors='coerce')
    a = a.dropna(subset=['alpha', '年季'])
    return a.rename(columns={'證券代碼': '基金代碼'})[['基金代碼', '年季', 'alpha']]

def extract_code_token(s: str):
    s = str(s).strip().upper()
    if not s:
        return ""
    tok = s.split()[0].split('-')[0]
    return re.sub(r'[^0-9A-Z]', '', tok)

def prep_holding_from_fund_data(fund_data: pd.DataFrame, 
                                fund_col='證券代碼', q_col='年季',
                                stock_id='標的碼', weight_col='投資比率％',
                                keep_asset_col='投資標的', keep_asset_value='股票型'):
    h = fund_data.copy()
    if keep_asset_col in h.columns:
        h = h[h[keep_asset_col].astype(str).str.contains(str(keep_asset_value))].copy()
    h['基金代碼'] = h[fund_col].astype(str).str.strip()
    h['年季'] = pd.PeriodIndex(h[q_col].astype(str).str.strip(), freq='Q')
    h['key_code'] = h[stock_id].astype(str).apply(extract_code_token)
    h['w_raw'] = pd.to_numeric(h[weight_col], errors='coerce') / 100
    h = h.groupby(['基金代碼', '年季', 'key_code'], as_index=False).agg(w_raw=('w_raw', 'sum'))
    h = h[h['w_raw'] > 0].copy()
    h['w'] = h['w_raw'] / h.groupby(['基金代碼', '年季'])['w_raw'].transform('sum')
    return h[['基金代碼', '年季', 'key_code', 'w']]

# -----------------------------
# Stock Measure Quality (Cohen et al. 2005)
# -----------------------------
def compute_stock_measure_quality(alpha_q: pd.DataFrame, holding_w: pd.DataFrame,
                      min_funds: int = 10, winsor: float | None = None):
    """
    計算 Stock Measure Quality (Cohen et al. 2005)。
    其實就是持股加權平均的基金 Alpha。
    """
    A = alpha_q.copy()
    H = holding_w.copy()
    out = []
    common_quarter = sorted(set(A['年季']) & set(H['年季']))
    for q in common_quarter:
        a_q = A[A['年季'].eq(q)][['基金代碼', 'alpha']].dropna().copy()
        
        # 對 Alpha 進行去極值 (可選)
        if winsor is not None and 0 < winsor < 0.5 and len(a_q) >= 5:
            lo = a_q['alpha'].quantile(winsor)
            hi = a_q['alpha'].quantile(1 - winsor)
            if pd.notna(lo) and pd.notna(hi) and lo <= hi:
                a_q.loc[:, 'alpha'] = a_q['alpha'].clip(lower=lo, upper=hi)
        
        h_q = H[H['年季'].eq(q)][['基金代碼', 'key_code', 'w']].copy()
        h_q = h_q[h_q['基金代碼'].isin(a_q['基金代碼'])]
        
        if len(a_q) < min_funds:
            continue
            
        merged = pd.merge(h_q, a_q, on='基金代碼', how='inner')
        if merged.empty:
            continue
        
        # Cohen et al. (2005) 的核心公式：
        # Stock Quality = Sum(w_ij * Alpha_j) / Sum(w_ij)
        # 這裡的 w 已經在 prep_holding 時歸一化為基金內的權重，但在這裡聚合時
        # 分母 Sum(w_ij) 代表這檔股票被多少權重的基金持有。
        # 不過通常實作上就是 加權平均 Alpha。
        
        smq = (
            merged.groupby('key_code', as_index=False)
                  .apply(lambda x: pd.Series({'SMQ': np.average(x['alpha'], weights=x['w'])
                                               if np.sum(x['w']) > 0 else np.nan}))
        )
        smq['年季'] = q
        smq['n_funds'] = len(a_q)
        out.append(smq[['年季','key_code','SMQ','n_funds']])
        
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['年季','key_code','SMQ','n_funds'])

# -----------------------------
# 股票 & 回測 +【持有期起始價資格】
# -----------------------------
def prep_stock_monthly_for_backtest(df_stock: pd.DataFrame,
                                    code_col='證券代碼', date_col='年月',
                                    ret_pct_col='報酬率％_月', price_col='收盤價(元)_月'):
    df = df_stock.copy()
    part = df[code_col].astype(str).str.strip().str.split(r'\s+', n=1, expand=True)
    df['標的碼'] = part[0]
    df['key_code'] = df['標的碼'].apply(extract_code_token)
    df['年月'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    df['ret_month'] = pd.to_numeric(df[ret_pct_col], errors='coerce') / 100
    df['price_month_end'] = pd.to_numeric(df.get(price_col, np.nan), errors='coerce')
    return df[['key_code', '年月', 'ret_month', 'price_month_end']].dropna(subset=['年月', 'key_code'])

def trim_outliers_cross_section(df: pd.DataFrame, value_col: str, time_col: str, p: float = 0.01, min_obs: int = 20):
    if p is None or p <= 0 or p >= 0.5:
        return df.copy()
    def _trim_group(g):
        if len(g) < min_obs:
            return g
        ql = g[value_col].quantile(p)
        qh = g[value_col].quantile(1 - p)
        if pd.isna(ql) or pd.isna(qh) or ql > qh:
            return g
        return g[(g[value_col] >= ql) & (g[value_col] <= qh)]
    return df.groupby(time_col, group_keys=False).apply(_trim_group).reset_index(drop=True)

def monthly_to_quarter_return(df_monthly: pd.DataFrame):
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q')
    qret = d.groupby(['key_code', '年季'], as_index=False).agg(q_ret=('ret_month', lambda s: (1+s).prod()-1),
                                                               n_months=('ret_month','count'))
    return qret[qret['n_months']==3].reset_index(drop=True)

def build_holding_start_eligibility(stock_m_with_price: pd.DataFrame, min_price: float):
    sm = stock_m_with_price.copy()
    sm['年季'] = sm['年月'].dt.to_period('Q')
    first_month = (sm.groupby(['key_code','年季'])['年月'].transform('min') == sm['年月'])
    first_rows = sm[first_month].copy()
    first_rows['eligible'] = (first_rows['price_month_end'] >= float(min_price)).astype(int)
    return first_rows[['key_code','年季','eligible']]

# -----------------------------
# Newey-West & 回測（百分比輸出）
# -----------------------------
def _newey_west_t(series, lags=NW_LAGS):
    y = pd.Series(series).dropna()
    if len(y) < 5:
        return np.nan, np.nan, len(y)
    X = np.ones((len(y), 1))
    res = sm.OLS(y.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    mean, se = res.params[0], res.bse[0]
    tval = mean / se if se > 0 else np.nan
    return float(mean), float(tval), float(len(y))

def backtest_single_decile(gia_df: pd.DataFrame, qret_df: pd.DataFrame,
                           score_col='SMQ', n_group=10, nw_lags=NW_LAGS,
                           value_fmt="{mean_pct:.2f}% ({t:.2f})", return_ts=False,
                           eligibility_df: pd.DataFrame | None = None):
    g = gia_df.copy()
    if not pd.api.types.is_period_dtype(g['年季']):
        g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq: pd.DataFrame):
        dfq = dfq.copy()
        dfq['group'] = pd.qcut(dfq[score_col].rank(method='first'),
                               q=n_group, labels=False, duplicates='drop') + 1
        return dfq

    g_grp = g.groupby('年季', group_keys=False).apply(assign_groups).reset_index(drop=True)
    g_grp['持有季'] = g_grp['年季'] + 1

    merged = pd.merge(
        g_grp, qret_df,
        left_on=['key_code', '持有季'],
        right_on=['key_code', '年季'],
        how='left', suffixes=('', '_ret')
    )

    if eligibility_df is not None and not eligibility_df.empty:
        merged = merged.merge(
            eligibility_df.rename(columns={'年季':'持有季'}),
            on=['key_code','持有季'],
            how='left'
        )
        merged = merged[merged['eligible'].fillna(0).eq(1)].copy()
        merged.drop(columns=['eligible'], inplace=True, errors='ignore')

    port = (merged.dropna(subset=['q_ret'])
                  .groupby(['年季', 'group'], as_index=False)
                  .agg(ret_mean=('q_ret', 'mean')))

    wide = port.pivot(index='年季', columns='group', values='ret_mean').sort_index()
    for k in range(1, n_group+1):
        if k not in wide.columns:
            wide[k] = np.nan
    wide = wide[sorted(wide.columns)]
    wide['long_short'] = wide[n_group] - wide[1]

    rows = []
    for col in list(range(1, n_group+1)) + ['long_short']:
        m, t, T = _newey_west_t(wide[col], lags=nw_lags)
        rows.append({'portfolio': col, 'mean': m, 'mean_pct': m*100, 't': t, 'T': T})
    summary = pd.DataFrame(rows).set_index('portfolio')
    table = summary.apply(lambda r: value_fmt.format(mean_pct=r['mean_pct'], t=r['t']),
                          axis=1).to_frame('Q1_only')

    return (table, wide) if return_ts else table

# -----------------------------
# 绩效指標（百分比顯示）
# -----------------------------
def portfolio_metrics_from_wide(wide: pd.DataFrame, freq: str = 'Q') -> pd.DataFrame:
    ann_map = {'Q': 4, 'M': 12}
    ann_factor = ann_map.get(freq.upper(), 4)

    cols = [c for c in wide.columns if (isinstance(c, (int, np.integer)) or c == 'long_short')]
    rows = []
    for c in cols:
        r = pd.to_numeric(wide[c], errors='coerce').dropna()
        T = int(r.shape[0])
        if T == 0:
            rows.append({'portfolio': c, 'mean_pct': np.nan, 'std_pct': np.nan,
                         'mean_to_vol': np.nan, 'ann_sharpe': np.nan,
                         'T': 0, 'hit_ratio_pct': np.nan})
            continue
        mean = float(r.mean())
        std = float(r.std(ddof=1)) if T > 1 else np.nan
        mtv = (mean / std) if (std is not None and std > 0) else np.nan
        ann_sharpe = (mtv * np.sqrt(ann_factor)) if np.isfinite(mtv) else np.nan
        hit_ratio = float((r > 0).mean()) * 100.0

        rows.append({'portfolio': c,
                     'mean_pct': mean * 100.0,
                     'std_pct' : std * 100.0 if std == std else np.nan,
                     'mean_to_vol': f"{mtv:.3f}",
                     'ann_sharpe': f"{ann_sharpe:.3f}",
                     'T': T,
                     'hit_ratio_pct': hit_ratio})
    out = pd.DataFrame(rows).set_index('portfolio')
    def _fmt_pct(x): 
        return (f"{x:.2f}%" if pd.notna(x) else "")
    out_print = out.copy()
    out_print['mean_pct'] = out['mean_pct'].map(_fmt_pct)
    out_print['std_pct']  = out['std_pct'].map(_fmt_pct)
    out_print['hit_ratio_pct'] = out['hit_ratio_pct'].map(_fmt_pct)
    return out_print[['mean_pct', 'std_pct', 'mean_to_vol', 'ann_sharpe', 'T', 'hit_ratio_pct']]

# -----------------------------
# 主程式：單次跑（無 grid search）
# -----------------------------
def main():
    print("正在讀取資料...")
    df_fund = pd.read_csv("fund_data/merged_fund_data.csv")
    df_factor = pd.read_csv("fund_data/carhart_factor.csv", encoding='UTF-16 LE', sep='\t')
    df_holding = pd.read_csv("fund_data/fund_data.csv")
    df_stock = pd.read_csv("fund_data/stock_return.csv", encoding='UTF-16 LE', sep='\t')

    # 1) 基金 α（CAPM）
    print("計算 CAPM Alpha...")
    fund_data = prepare_fund_data(df_fund)
    # 準備 CAPM 因子
    factor_data = prepare_factor_data(df_factor) 
    merged = merge_fund_factor(fund_data, factor_data)
    
    # 使用 CAPM 回歸
    coef = run_capm_rolling(merged)
    coef_clean = clean_alpha_panel(coef)
    alpha = prepare_fund_alpha(coef_clean)

    # 2) 基金持股
    print("處理基金持股...")
    holding = prep_holding_from_fund_data(df_holding)

    # 3) 股票報酬與資格
    print("處理股票月/季報酬...")
    stock_m_all = prep_stock_monthly_for_backtest(df_stock)
    elig_hold = build_holding_start_eligibility(stock_m_all, min_price=MIN_PRICE)

    stock_m_use = trim_outliers_cross_section(
        stock_m_all, value_col='ret_month', time_col='年月', p=TRIM_P, min_obs=20
    )
    stock_q = monthly_to_quarter_return(stock_m_use)
    if stock_q.empty:
        print("季報酬為空：請檢查 stock_return 檔或資料覆蓋度")
        return

    # 4) Stock Measure Quality (Cohen et al. 2005)
    print("計算 Stock Measure Quality (SMQ)...")
    smq_df = compute_stock_measure_quality(alpha, holding, min_funds=10, winsor=None)
    if smq_df.empty:
        print("SMQ 為空：請檢查基金 α 與持股覆蓋度")
        return

    # 5) 十等分回測
    print("執行回測...")
    result_tbl, wide = backtest_single_decile(
        smq_df, stock_q, score_col='SMQ', n_group=10, nw_lags=NW_LAGS, eligibility_df=elig_hold, return_ts=True
    )
    
    print(f"\n=== Stock Measure Quality (CAPM) 十等分回測（MIN_PRICE={MIN_PRICE}）===\n")
    print(result_tbl)

    # 6) 投組績效
    perf = portfolio_metrics_from_wide(wide, freq='Q')
    order_idx = [c for c in perf.index if c != 'long_short'] + ['long_short']
    perf = perf.loc[order_idx]
    print("\n=== 投組績效指標（季頻；百分比顯示）===\n")
    print(perf)

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