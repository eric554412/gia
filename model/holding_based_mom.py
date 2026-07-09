import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
from tqdm import tqdm


MIN_PRICE = 10
NW_LAGS = 4
K_RATIO_LIST = [0.95]
K = 6

def to_month_end(s):
    '''將各種日期格式轉成月末日期'''
    s_str = s.astype(str)
    d1 = pd.to_datetime(s_str, format = '%Y%m', errors = 'coerce')
    d2 = pd.to_datetime(s_str, format = '%Y-%m', errors = 'coerce')
    d3 = pd.to_datetime(s_str, errors = 'coerce')
    
    d = d1.fillna(d2).fillna(d3)
    return d.dt.to_period('M').dt.to_timestamp('M')


def extract_code_token(s: str):
    '''把證券代碼字串標準化，取出主要代碼部分'''
    s = str(s).strip().upper()
    if not s: 
        return ""
    
    tok = s.split()[0].split('-')[0]
    return re.sub(r'[^A-Z0-9]', '', tok)


def prep_fund_data(df_fund: pd.DataFrame, code_col = '證券代碼', date_col = '年月',
                   ret_col = '單月ROI', pct_as_percent = True):
    '''準備基金月報酬資料，標準化欄位名稱與格式'''
    df = df_fund.copy()
    df[code_col] = df[code_col].astype(str).str.strip()
    df[date_col] = to_month_end(df[date_col])
    df[ret_col] = pd.to_numeric(df[ret_col], errors = 'coerce')    
    if pct_as_percent: 
        df[ret_col] /= 100
    return df[[code_col, date_col, ret_col]].rename(columns = {code_col: '基金代碼', ret_col: 'ret_month'})


def prep_holding_from_fund_data(fund_data: pd.DataFrame, fund_col = '證券代碼', q_col = '年季', stock_id = '標的碼', weight_col = '投資比率％',
                                keep_asset_col = '投資標的', keep_asset_value = '股票型'):
    '''從基金資料中整理出持股資料，並將持股權重標準化'''
    h = fund_data.copy()
    if keep_asset_col in h.columns:
        h = h[h[keep_asset_col].astype(str).str.contains(str(keep_asset_value), na = False)]
    h['基金代碼'] = h[fund_col].astype(str).str.strip()
    h['年季'] = pd.PeriodIndex(h[q_col].astype(str).str.strip(), freq = 'Q')
    h['key_code'] = h[stock_id].astype(str).apply(extract_code_token)
    h['w_raw'] = pd.to_numeric(h[weight_col], errors = 'coerce') / 100
    h = h.groupby(['基金代碼', '年季', 'key_code'], as_index = False).agg(w_raw = ('w_raw', 'sum'))
    h = h[h['w_raw'] > 0].copy()
    h['w'] = h['w_raw'] / h.groupby(['基金代碼', '年季'])['w_raw'].transform('sum')
    return h[['基金代碼', '年季', 'key_code', 'w']]


def prep_stock_monthly_for_backtest(df_stock: pd.DataFrame, code_col = '證券代碼', date_col = '年月',
                                    ret_pct_col = '報酬率％_月', price_col = '收盤價(元)_月'):
    '''準備股票月報酬資料，標準化欄位名稱與格式'''
    df = df_stock.copy()
    part = df[code_col].astype(str).str.strip().str.split(r'\s+', n = 1, expand = True)
    df['key_code'] = part[0].apply(extract_code_token)
    df['年月'] = pd.to_datetime(df[date_col].astype(str), format = '%Y%m', errors = 'coerce').dt.to_period('M').dt.to_timestamp('M')
    df['ret_month'] = pd.to_numeric(df[ret_pct_col], errors = 'coerce') / 100
    df['price_month_end'] = pd.to_numeric(df.get(price_col, np.nan), errors = 'coerce')
    return df[['key_code', '年月', 'ret_month', 'price_month_end']].dropna(subset = ['ret_month'])


def build_entry_eligibility(stock_m: pd.DataFrame, min_price):
    '''季末篩選股票, 價格需高於 min_price 才有資格進入下一季投資組合'''
    sm = stock_m.copy()
    sm['年季'] = sm['年月'].dt.to_period('Q')
    sm['is_qe'] = sm['年月'].dt.is_quarter_end
    qe = sm[sm['is_qe']].copy()
    qe['持有季'] = (qe['年季'] + 1).astype('period[Q]')
    qe['eligible'] = (qe['price_month_end'] >= min_price).astype(int)
    return qe[['key_code', '持有季', 'eligible']]    


def monthly_to_quarterly_return(df_monthly: pd.DataFrame, key_col = 'key_code', ret_col = 'ret_month'):
    '''將月報酬資料轉成季報酬資料'''
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q')
    qret = d.groupby([key_col, '年季'], as_index = False).agg(
        q_ret = (ret_col, lambda x: (1 + x).prod() - 1),
        n_months = (ret_col, 'count')
    )
    return qret[qret['n_months'] == 3].reset_index(drop = True)


def prepare_factor_data(df_factor: pd.DataFrame):
    '''讀取並標準化 Carhart 四因子資料'''
    fac = df_factor.copy()
    fac['年月'] = to_month_end(fac['年月'])
    mapping = {
        "市場風險溢酬": "MKT",
        "規模溢酬 (3因子)": "SMB",
        "淨值市價比溢酬": "HML",
        "動能因子": "MOM",
        "無風險利率": "RF_annual"
    }
    fac = fac.rename(columns = mapping)
    for col in ["MKT", "SMB", "HML", "MOM"]:
        fac[col] = pd.to_numeric(fac[col], errors = 'coerce') / 100
    # 將年化無風險利率轉成月度
    fac["RF"] = (pd.to_numeric(fac["RF_annual"], errors = 'coerce') / 100) / 12
    return fac[['年月', 'MKT', 'SMB', 'HML', 'MOM', 'RF']]
    

def compute_hbm_skill(holding_df: pd.DataFrame, stock_monthly_df: pd.DataFrame, k = K):
    '''計算持倉動能，是基於持倉權重與股票動能的乘積和，是衡量存量，代表目前的持股強不強'''
    s_data = stock_monthly_df.sort_values(['key_code', '年月']).copy()
    s_data['ret_plus_1'] = s_data['ret_month'] + 1
    s_data['mom6'] = s_data.groupby('key_code')['ret_plus_1'].transform(
        lambda x: x.rolling(k).apply(np.prod, raw = True) - 1
    )
    s_mom = s_data[s_data['年月'].dt.is_quarter_end][['key_code', '年月', 'mom6']]
    s_mom['年季'] = s_mom['年月'].dt.to_period('Q')
    merged = pd.merge(holding_df, s_mom, on = ['key_code', '年季'], how = 'inner')
    hbm_skill = merged.groupby(['基金代碼', '年季']).apply(
        lambda x: np.sum(x['w'] * x['mom6'])
    ).reset_index(name = 'alpha')
    return hbm_skill

def compute_gia(alpha_q, holding_w, k_ratio = 0.5, min_funds = 10):
    '''計算 GIA（基於基金 alpha 的股票層級貢獻）'''
    def _truncated_pinv(W, k):
        U, s, Vt = np.linalg.svd(W, full_matrices = False)
        k = max(1, min(int(np.floor(k)), len(s)))
        return (Vt[:k].T / s[:k]) @ U[:, :k].T
    A = alpha_q.copy()
    H = holding_w.copy()
    out = []
    common_quarters = sorted(set(A['年季']) & set(H['年季']))
    for q in common_quarters:
        a_q = A.loc[A['年季'] == q, ['基金代碼', 'alpha']].dropna()
        h_q = H.loc[H['年季'] == q, ['基金代碼', 'key_code', 'w']]
        h_q = h_q[h_q['基金代碼'].isin(a_q['基金代碼'])].copy()
        funds = a_q['基金代碼'].unique()
        M = len(funds)
        if M < min_funds:
            continue
        stocks = h_q['key_code'].unique()
        N = len(stocks)
        if N == 0:
            continue
        f_id = {f:i for i, f in enumerate(funds)}
        s_id = {s:i for i, s in enumerate(stocks)}
        W = np.zeros((M, N))
        for _, row in h_q.iterrows():
            W[f_id[row['基金代碼']], s_id[row['key_code']]] = row['w']
        S = a_q.set_index('基金代碼').reindex(funds)['alpha'].to_numpy(float)
        K = max(1, int(np.floor(k_ratio * M)))
        alpha_stock = _truncated_pinv(W, K) @ S
        out.append(pd.DataFrame({'年季': q, 'key_code': stocks, 'GIA': alpha_stock}))
    return pd.concat(out, ignore_index = True) if out else pd.DataFrame()


def _newey_west_t(series, lags = 6):
    '''計算 Newey-West t 統計量'''
    y = pd.Series(series).dropna()
    if len(y) < 5:
        return np.nan, np.nan, len(y)
    res = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type = 'HAC', cov_kwds = {'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0]), len(y)


def backtest_single_decile(
    gia_df: pd.DataFrame,
    qret_df: pd.DataFrame,
    eligibility_df: pd.DataFrame = None,
    gia_col='GIA',
    n_groups=10,
    nw_lags=NW_LAGS
):
    '''對 GIA 排序後進行投資組合回測'''
    g = gia_df.copy()
    if not pd.api.types.is_period_dtype(g['年季']):
        g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq):
        dfq = dfq.copy()
        dfq['decile'] = pd.qcut(
            dfq[gia_col].rank(method='first'),
            q=n_groups,
            labels=False,
            duplicates='drop'
        ) + 1
        return dfq

    g_grp = g.groupby('年季', group_keys=False).apply(assign_groups).reset_index(drop=True)
    g_grp = g_grp.rename(columns={'年季': 'formation_q'})
    g_grp['持有季'] = g_grp['formation_q'] + 1
    
    if eligibility_df is not None:
        elig = eligibility_df.copy()
        elig['持有季'] = pd.PeriodIndex(elig['持有季'], freq = 'Q')
        g_grp = g_grp.merge(elig, on=['key_code', '持有季'], how='left')
        g_grp = g_grp[g_grp['eligible'].fillna(0).eq(1)]
    
    qret = qret_df.copy()
    if not pd.api.types.is_period_dtype(qret['年季']):
        qret['年季'] = pd.PeriodIndex(qret['年季'], freq='Q')
        

    merged = pd.merge(
        g_grp,
        qret,
        left_on=['key_code', '持有季'],
        right_on=['key_code', '年季'],
        how='left'
    )

    port = (merged.dropna(subset=['q_ret'])
                  .groupby(['formation_q', 'decile'], as_index=False)
                  .agg(ret_mean=('q_ret', 'mean')))

    wide = port.pivot(index='formation_q', columns='decile', values='ret_mean').sort_index()
    for k in range(1, n_groups + 1):
        if k not in wide.columns:
            wide[k] = np.nan
    wide = wide[sorted(wide.columns)]
    wide['long_short'] = wide[n_groups] - wide[1]

    rows = []
    for col in list(range(1, n_groups + 1)) + ['long_short']:
        m, t, T = _newey_west_t(wide[col], lags=nw_lags)
        rows.append({'portfolio': col, 'mean': m, 't': t})
    summary = pd.DataFrame(rows).set_index('portfolio')

    return wide, summary



def calc_monotonicity(wide, n_groups = 10):
    '''計算投資組合報酬的單調性指標'''
    cols = list(range(1, n_groups + 1))
    mean_rets = wide[cols].mean()
    ranks = pd.Series(range(1, n_groups + 1), index = cols)
    spearman_rho = mean_rets.corr(ranks, method = 'spearman')
    rets_list = mean_rets.values
    violation = 0
    for i in range(len(rets_list) - 1):
        if rets_list[i] > rets_list[i + 1]:
            violation += 1
    return spearman_rho, violation


def build_slim_metrics_table(wide, summary_raw, periods_per_years = 4):
    '''建立績效表'''
    cols = [*sorted(c for c in wide.columns if isinstance(c, int)), 'long_short']
    out = []
    for col in cols:
        r = wide[col] if col in wide.columns else pd.Series(dtype = float)
        if r.empty:
            mean_q = std_q = mtv = sharpe_ann = np.nan
        else:
            mean_q = r.mean()
            std_q = r.std()
            mtv = np.nan if (pd.isna(std_q) or std_q == 0) else mean_q / std_q # 未年化的 sharpe ratio
            sharpe_ann = np.nan if (pd.isna(std_q) or std_q == 0) else mtv * np.sqrt(periods_per_years)
        tval = summary_raw.loc[col, 't'] if col in summary_raw.index else np.nan
        out.append([col, mean_q * 100, std_q * 100, mtv, sharpe_ann, tval])
    
    slim_num = pd.DataFrame(out, columns = ['portfolio', 'mean_pct', 'std_pct', 'mean_to_vol', 'sharpe_annual', 't值']).set_index('portfolio')
    fmt_pct = lambda x: "" if pd.isna(x) else f"{x:.2f}%"
    fmt_val = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    fmt_t = lambda x: "" if pd.isna(x) else f"{x:.2f}"
    
    slim_fmt = pd.DataFrame({
        'mean_pct': slim_num['mean_pct'].apply(fmt_pct),
        'std_pct': slim_num['std_pct'].apply(fmt_pct),
        'mean_to_vol': slim_num['mean_to_vol'].apply(fmt_val),
        'sharpe_annual': slim_num['sharpe_annual'].apply(fmt_val),
        't值': slim_num['t值'].apply(fmt_t)
    })
    return slim_num, slim_fmt


def main():
    print(f"===== 啟動最佳化回測 =====")
    try:
        print("讀取檔案...")
        df_holding = pd.read_csv("fund_data/fund_data.csv", encoding = 'utf-8')
        df_stock = pd.read_csv("fund_data/stock_return.csv", encoding = 'utf-16', sep = '\t')
        df_factor = pd.read_csv("fund_data/carhart_factor.csv", encoding = 'utf-16', sep = '\t')
    except Exception as e:
        print(f"讀取檔案失敗: {e}")
        return
    
    print("資料前處理")
    factor_data = prepare_factor_data(df_factor)
    holding = prep_holding_from_fund_data(df_holding)
    holding['年季'] = holding['年季'] + 1
    stock_m = prep_stock_monthly_for_backtest(df_stock)
    stock_q = monthly_to_quarterly_return(stock_m)
    entry_elig = build_entry_eligibility(stock_m, MIN_PRICE)
    
    if stock_q.empty:
        print("錯誤: 股票資料為空")
        return
    
    print("計算持倉alpha")
    fund_skill = compute_hbm_skill(holding, stock_m)
    print(f"搜尋最佳化參數(K範圍: {K_RATIO_LIST[0]} ~ {K_RATIO_LIST[-1]})...")
    results = []
    for k in tqdm(K_RATIO_LIST, desc="K_RATIO 掃描"):
        gia = compute_gia(fund_skill, holding, k_ratio = k)
        if gia.empty: 
            continue
        wide, summary = backtest_single_decile(gia, stock_q, entry_elig, n_groups=10)
        rho, viol = calc_monotonicity(wide, n_groups = 10)
        ls_t = summary.loc['long_short', 't']
        results.append({
            'mean': summary.loc['long_short', 'mean'],
            'k': k,
            'rho': rho,
            'viol': viol,
            't': ls_t
        })
    
    df_res = pd.DataFrame(results).sort_values(by = ['rho', 'mean', 't', 'viol'], ascending = [True, False, False, False])
    best_k = df_res.iloc[0]['k']
    best_rho = df_res.iloc[0]['rho']
    
    print("\n" + "="*50)
    print(f"【最佳參數確認】")
    print(f"  > Best K_RATIO : {best_k:.2f}")
    print(f"  > Spearman Rho : {best_rho:.4f}")
    print("="*50 + "\n")
    print("進行最終回測...")
    final_gia = compute_gia(fund_skill, holding, k_ratio = best_k)
    final_wide, final_summary = backtest_single_decile(final_gia, stock_q, entry_elig, gia_col='GIA', n_groups = 10, nw_lags = NW_LAGS)
    _, slim_fmt = build_slim_metrics_table(final_wide, final_summary)
    
    print("\n" + "="*60)
    print(f"=== 最終績效報表 (K={best_k:.2f}, Price={MIN_PRICE}) ===")
    print("="*60)
    print(slim_fmt)
    print("="*60)

if __name__ == "__main__":
    main()
    