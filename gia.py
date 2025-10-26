import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
'''
複製
Forecasting Stock Returns through An Efficient Aggregation of Mutual Fund Holdings
改成台灣基金資料
'''

def to_month_end(s):
    '''將日期轉為月底'''
    s_str = s.astype(str)
    d1 = pd.to_datetime(s_str, format = '%Y%m', errors = 'coerce')
    d2 = pd.to_datetime(s_str, format = '%Y-%m', errors = 'coerce')
    d3 = pd.to_datetime(s_str, errors = 'coerce')
    d = d1.fillna(d2).fillna(d3)
    return d.dt.to_period('M').dt.to_timestamp('M')


def prepare_fund_data(df_fund: pd.DataFrame, code_col = '證券代碼', date_col = '年月', ret_col = '單月ROI', pct_as_percent = True):
    '''準備基金資料'''
    df = df_fund.copy()
    df[code_col] = df[code_col].astype(str).str.strip() # 證券代碼轉字串並去空白
    df[date_col] = to_month_end(df[date_col]) # 日期轉月底
    df[ret_col] = pd.to_numeric(df[ret_col], errors = 'coerce')
    if pct_as_percent:
        df[ret_col] = df[ret_col] / 100 # 百分比轉小數
    return df[[code_col, date_col, ret_col]]

def prepare_factor_data(df_factor: pd.DataFrame, date_col = '年月', mktrf_col = '市場風險溢酬', smb_col = '規模溢酬 (3因子)', 
                        hml_col = '淨值市價比溢酬', mom_col = '動能因子', rf_col = '無風險利率', pct_as_percent = True):
    '''準備因子資料'''
    fac = df_factor.copy()
    fac[date_col] = to_month_end(fac[date_col]) # 日期轉月底
    for c in [mktrf_col, smb_col, hml_col, mom_col, rf_col]:
        fac[c] = pd.to_numeric(fac[c], errors = 'coerce')
    if pct_as_percent:
        for c in [mktrf_col, smb_col, hml_col, mom_col]:
            fac[c] = fac[c] / 100 # 百分比轉小數
    # tej 的無風險利率是年化的，轉為月化
    rf_annual = fac[rf_col] / 100 if pct_as_percent else fac[rf_col]
    rf_monthly = rf_annual / 12
    fac['RF'] = rf_monthly
    fac = fac[[mktrf_col, smb_col, hml_col, mom_col, 'RF', date_col]].rename(
        columns = {mktrf_col: 'MKT', smb_col: 'SMB', hml_col: 'HML', mom_col: 'MOM'}
        )
    fac = fac.sort_values(by = [date_col]).drop_duplicates(subset = [date_col], keep = 'last').reset_index(drop = True)
    return fac


def merge_fund_factor(fund_df, fac_df, date_col = '年月', code_col = '證券代碼', ret_col = '單月ROI'):
    '''合併基金、因子資料'''
    merged = pd.merge(fund_df, fac_df, on = date_col, how = 'inner').sort_values([code_col, date_col]).reset_index(drop = True)
    merged['ret_excess'] = merged[ret_col] - merged['RF'] # 計算超額報酬
    return merged


def run_carhart_rolling(
    merged_df: pd.DataFrame, date_col = '年月', code_col = '證券代碼',
    y_col = 'ret_excess', x_cols = ('MKT', 'SMB', 'HML', 'MOM'),
    window = 18, min_periods = 18, cov_type = None):
    '''跑 Carhart 4 因子回歸， 每個季末跑一次回歸，使用過去12個月的資料'''
    
    df = merged_df.copy().sort_values(by = [code_col, date_col]).reset_index(drop = True)
    out_frames = []
    for fid, g in df.groupby(code_col, sort = False):
        # 用基金分組跑回歸
        g = g.reset_index(drop = True)
        y_all = g[y_col].to_numpy(float)
        X_all = g[list(x_cols)].to_numpy(float) # n * k array
        n = len(g)
        # 裝參數的容器
        alpha = np.full(n, np.nan)
        betas = np.full((n, len(x_cols)), np.nan)
        r2s = np.full(n, np.nan)
        nobs = np.zeros(n, int)
        qe_mask = g[date_col].dt.is_quarter_end.to_numpy() # 檢查是否是季末
        qe_idx = np.flatnonzero(qe_mask) # 季末的索引
        for t in qe_idx:
            # 如果不足12個月就跳過
            end = t + 1
            start = max(0, end - window)
            if end - start < min_periods:
                continue
            y_win = y_all[start:end]
            X_win = X_all[start:end, :]
            mask = np.isfinite(y_win) & np.all(np.isfinite(X_win), axis = 1)
            if mask.sum() < min_periods:
                continue
            y_use = y_win[mask]
            X_use = pd.DataFrame(X_win[mask], columns = list(x_cols))
            X_use = sm.add_constant(X_use, has_constant = 'add')
            
            model = sm.OLS(y_use, X_use, missing = 'drop')
            res = model.fit(cov_type = cov_type) if cov_type else model.fit() 
            alpha[t] = res.params['const'] if 'const' in res.params.index else np.nan
            for i, c in enumerate(x_cols):
                betas[t, i] = res.params[c] if c in res.params.index else np.nan
            r2s[t] = res.rsquared
            nobs[t] = int(res.nobs)
        sel = qe_mask
        out = pd.DataFrame({
            date_col: g.loc[sel, date_col].to_numpy(),
            code_col: g.loc[sel, code_col].to_numpy(),
            'alpha': alpha[sel],
            **{f'beta_{c.lower()}': betas[sel, i] for i, c in enumerate(x_cols)},
            'r2': r2s[sel],
            'n_obs': nobs[sel]})
        out['valid_alpha'] = (out['n_obs'] >= min_periods).astype(int)
        out_frames.append(out)
    coef_df = pd.concat(out_frames, ignore_index = True)
    return coef_df


def clean_alpha_panel(df: pd.DataFrame, date_col = '年月', code_col = '證券代碼', alpha_col = 'alpha',
                      beta_prefix = 'beta_', factors = ('mkt', 'smb', 'hml', 'mom'), r2_col = 'r2',
                      nobs_col = 'n_obs', valid_col = 'valid_alpha', dropna_alpha = True, sort = True):
    '''清理 alpha panel 資料'''
    out = df.copy()
    out = out[out[valid_col].eq(1)].copy() # 只保留 valid alpha
    beta_cols = [f'{beta_prefix}{f}' for f in factors if f'{beta_prefix}{f}' in out.columns] # 確認 beta 欄位存在
    cols = [code_col, date_col, alpha_col] + beta_cols + [r2_col, nobs_col, valid_col]
    out = out[cols]
    if dropna_alpha and alpha_col in out.columns:
        out = out.dropna(subset = [alpha_col])
    if sort:
        out = out.sort_values(by = [code_col, date_col]).reset_index(drop = True)
    return out


def extract_code_token(s: str):
    '''從股票代碼抽出數字部分'''
    s = str(s).strip().upper()
    if not s:
        return ""
    tok = s.split()[0] # 取第一個空白分隔部分
    tok = tok.split('-')[0] # 如 3664-KY 變3664
    tok = re.sub(r'[^0-9A-Z]', '', tok)
    return tok


def to_quarter_end(s):
    '''轉換成季末 timestamp'''
    s = pd.Series(s)
    raw = s.copy()
    # 先用一般方法解析
    dt = pd.to_datetime(raw, errors = 'coerce')
    # 處理 'YYYYQn'
    mask = dt.isna()
    if mask.any():
        cand = raw[mask].astype(str).str.strip().str.upper()
        is_q = cand.str.match(r'^\d{4}Q[1-4]$')
        if is_q.any():
            idx = cand.index[is_q]
            dt.loc[idx] = pd.PeriodIndex(cand.loc[idx], freq = 'Q').to_timestamp('Q')
    return dt.dt.to_period('Q').dt.to_timestamp('Q')


def prepare_fund_alpha(alpha_df: pd.DataFrame, fund_col = '證券代碼', date_col = '年月',
                       alpha_col = 'alpha', valid_col = 'valid_alpha'):
    '''準備 fund_alpha 資料清理'''
    a = alpha_df.copy()
    if valid_col in a.columns:
        a = a[a[valid_col].eq(1)].copy()
    a['年季'] = to_quarter_end(a[date_col])
    a[alpha_col] = pd.to_numeric(a[alpha_col], errors = 'coerce')
    a = a.dropna(subset = [alpha_col, '年季'])
    a['年季'] = a['年季'].dt.to_period('Q')
    return a.rename(columns = {fund_col: '基金代碼', alpha_col: 'alpha'})[['基金代碼', '年季', 'alpha']]


def prep_holding_from_fund_data(fund_data: pd.DataFrame, fund_col = '證券代碼', q_col = '年季',
                                stock_id = '標的碼', weight_col = '投資比率％', keep_asset_col = '投資標的',
                                keep_asset_value = '股票型'):
    '''準備 fund_holding 資料清理'''
    h = fund_data.copy()
    if keep_asset_col in fund_data.columns:
        h = h[h[keep_asset_col].astype(str).str.contains(str(keep_asset_value))].copy()
    h['基金代碼'] = h[fund_col].astype(str).str.strip()
    # 統一季別為 Period['Q']
    h['年季'] = pd.PeriodIndex(h[q_col].astype(str).str.strip(), freq = 'Q')
    h['key_code'] = h[stock_id].astype(str).apply(extract_code_token)
    # 權重百分比到小數
    h['w_raw'] = pd.to_numeric(h[weight_col], errors = 'coerce') / 100
    h = h.groupby(['基金代碼', '年季', 'key_code'], as_index = False).agg(w_raw = ('w_raw', 'sum'))
    h = h[h['w_raw'] > 0].copy()
    h['w'] = h['w_raw'] / h.groupby(['基金代碼', '年季'])['w_raw'].transform('sum')
    h['w'] = h['w'].fillna(0)
    return h[['基金代碼', '年季', 'key_code', 'w']]


def _truncated_pinv(W: np.ndarray, k: int):
    '''截斷式廣義逆矩陣, k用論文設定基金數的一半'''
    U, s, Vt = np.linalg.svd(W, full_matrices = False) # svd分解, 這邊的Vt為轉置後的, 後面要轉回來
    k_max = int((s > 0).sum())
    k = max(1, min(int(k), k_max))
    U_k, s_k, Vt_k = U[:, :k], s[:k], Vt[:k, :]
    return (Vt_k.T / s_k) @ U_k.T


def compute_gia(alpha_q: pd.DataFrame, 
                holding_w: pd.DataFrame, 
                lag_holidings = 0, # 0用GIA, 1用LGIA
                k_ratio = 0.5,
                min_funds = 10):
    '''開始計算 GIA'''
    A = alpha_q.copy()
    H = holding_w.copy()
    if lag_holidings == 1:
        H['年季'] = (H['年季'] + 1).astype('period[Q]')
    out = []
    common_quarter = sorted(set(A['年季']) & set(H['年季']))
    for q in common_quarter:
        # 開始每季計算 GIA
        a_q = A.loc[A['年季'].eq(q), ['基金代碼', 'alpha']].dropna()
        h_q = H.loc[H['年季'].eq(q), ['基金代碼', 'key_code', 'w']]
        h_q = h_q[h_q['基金代碼'].isin(a_q['基金代碼'])].copy()
        funds = pd.Index(a_q['基金代碼'].unique())
        M = len(funds)
        if M < min_funds:
            continue
        stocks = pd.Index(h_q['key_code'].unique())
        N = len(stocks)
        if N == 0:
            continue
        W = np.zeros((M, N), dtype = float) # 建立權重矩陣
        f_id = pd.Series(range(M), index = funds)
        s_id = pd.Series(range(N), index = stocks)
        for _, row in h_q.iterrows():
            W[f_id[row['基金代碼']], s_id[row['key_code']]] = row['w']
        # 對齊W的基金順序
        S_hat = a_q.set_index('基金代碼').reindex(funds)['alpha'].to_numpy(float)
        # 廣義逆矩陣截斷到 K
        K = max(1, int(np.floor(k_ratio * M)))
        W_pinv = _truncated_pinv(W, K)
        alpha_stock = W_pinv @ S_hat
        
        label = 'GIA' if lag_holidings == 0 else 'LGIA'
        out.append(pd.DataFrame({
            '年季': q,
            'key_code': stocks,
            label: alpha_stock,
            'n_funds': M,
            'k_used': K 
        }))
    if not out:
        return pd.DataFrame(columns = ['年季', 'key_code', 'GIA' if lag_holidings == 0 else 'LGIA', 'n_funds', 'k_used'])
    return pd.concat(out, ignore_index = True)


def prep_stock_monthly_for_backtest(df_stock: pd.DataFrame, 
                                    code_col = '證券代碼', date_col = '年月',
                                    ret_pct_col = '報酬率％_月', price_col = '收盤價(元)_月',
                                    min_price: float | None = 10):
    '''準備股票資料清理, 踢掉股價小於10元的'''
    df = df_stock.copy()
    # 處理股票字串
    part = df[code_col].astype(str).str.strip().str.split(r'\s+', n = 1, expand = True)
    df['標的碼'] = part[0]
    df['標的名稱'] = part[1].fillna('')
    df['key_code'] = df['標的碼'].apply(extract_code_token)
    # 處理年月欄位
    m1 = pd.to_datetime(df[date_col].astype(str).str.strip(), format = '%Y%m', errors = 'coerce')
    df['年月'] = m1.dt.to_period('M').dt.to_timestamp('M')
    df['ret_month'] = pd.to_numeric(df[ret_pct_col] ,errors = 'coerce') / 100
    if (min_price is not None) and (price_col in df.columns):
        df[price_col] = pd.to_numeric(df[price_col], errors = 'coerce')
        df = df[df[price_col] >= min_price]
    df = df.dropna(subset = ['年月', 'key_code'])
    return df[['key_code', '標的碼', '標的名稱', '年月', 'ret_month']]


def monthly_to_quarter_return(df_monthly: pd.DataFrame):
    '''把月報酬聚合成季報酬'''
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q') # 對齊季度
    qret = d.groupby(['key_code', '年季'], as_index = False).agg(q_ret = ('ret_month', lambda s: (1 + s).prod() - 1),
                                                               n_months = ('ret_month', 'count'))
    qret = qret[qret['n_months'] == 3].copy()
    return qret.reset_index(drop = True)


def _newey_west_t(series, lags = 3):
    # 回歸 series = const + eps, 用HAC標準誤
    y = pd.Series(series).dropna()
    if len(y) < 5:
        return np.nan, np.nan, len(y)
    X = np.ones((len(y), 1))
    model = sm.OLS(y.values, X, missing = 'drop')
    res = model.fit(cov_type = 'HAC', cov_kwds = {'maxlags': lags})
    mean = res.params[0] # 樣本平均數(係數)
    se = res.bse[0] # HAC調整後的標準誤
    tval = mean / se if se > 0 else np.nan
    return float(mean), float(tval), float(len(y))


def backtest_single_decile(gia_df: pd.DataFrame, qret_df: pd.DataFrame,
                           gia_col='GIA', n_group=10, nw_lags=3,
                           value_fmt="{mean_pct:.2f} ({t:.2f})", return_ts=False):
    """每季重算十個 decile，持有下一季，含 long-short + benchmark"""
    g = gia_df.copy()
    if not pd.api.types.is_period_dtype(g['年季']):
        g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq: pd.DataFrame):
        dfq = dfq.copy()
        dfq['group'] = pd.qcut(
            dfq[gia_col].rank(method='first'),
            q=n_group, labels=False, duplicates='drop'
        ) + 1
        return dfq

    g_grp = g.groupby('年季', group_keys=False).apply(assign_groups).reset_index(drop=True)
    g_grp['持有季'] = g_grp['年季'] + 1

    merged = pd.merge(
        g_grp, qret_df,
        left_on=['key_code', '持有季'],
        right_on=['key_code', '年季'],
        how='left', suffixes=('', '_ret')
    )

    port = (merged.dropna(subset=['q_ret'])
                  .groupby(['年季', 'group'], as_index=False)
                  .agg(ret_mean=('q_ret', 'mean')))

    wide = port.pivot(index='年季', columns='group', values='ret_mean').sort_index()
    for k in range(1, n_group+1):
        if k not in wide.columns:
            wide[k] = np.nan
    wide = wide[sorted(wide.columns)]

    # Long–Short 組
    wide['long_short'] = wide[n_group] - wide[1]

    # Newey-West 檢定 + 格式化
    rows = []
    for col in list(range(1, n_group+1)) + ['long_short']:
        m, t, T = _newey_west_t(wide[col], lags=nw_lags)
        rows.append({'portfolio': col, 'mean': m, 'mean_pct': m*100, 't': t, 'T': T})
    summary = pd.DataFrame(rows).set_index('portfolio')
    table = summary.apply(
        lambda r: value_fmt.format(mean=r['mean'], mean_pct=r['mean_pct'], t=r['t'], T=r['T']),
        axis=1
    ).to_frame('Q1_only')

    if return_ts:
        return table, wide
    else:
        return table




def main():
    df_fund = pd.read_csv("merged_fund_data.csv")
    df_factor = pd.read_csv("carhart_factor.csv", encoding = 'UTF-16 LE', sep = '\t')
    df_holding = pd.read_csv("fund_data.csv")
    df_stock = pd.read_csv('stock_return.csv', encoding = 'UTF-16 LE', sep = '\t')
    fund_data = prepare_fund_data(df_fund)
    factor_data = prepare_factor_data(df_factor)
    merged = merge_fund_factor(fund_data, factor_data)
    result = run_carhart_rolling(merged)
    out = clean_alpha_panel(result)
    alpha = prepare_fund_alpha(out)
    holding_data = prep_holding_from_fund_data(df_holding)
    gia = compute_gia(alpha_q = alpha, holding_w = holding_data)
    stock = prep_stock_monthly_for_backtest(df_stock, min_price = 30)
    stock_q = monthly_to_quarter_return(stock)
    result = backtest_single_decile(gia, stock_q)
    print("\n=== Naïve Carhart 四因子（忽略基金相關性）十等分回測結果 ===\n")
    print(result)
    
    

if __name__ == "__main__":
    main()