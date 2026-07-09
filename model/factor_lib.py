import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
from tqdm import tqdm


MIN_PRICE = 10
NW_LAGS = 4
WINDOW = 36


# 資料清理函數
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
    

# 計算基金 alpha proxy 函數
def compute_holding_alpha_skill(holding_df: pd.DataFrame, stock_monthly_df: pd.DataFrame, 
                                 factor_df: pd.DataFrame, window = WINDOW):
    '''計算持倉 alpha 當作基金 alpha proxy'''
    print(f"正在計算 Holding Alpha (window = {window} 月)")
    alpha_results = []
    quarters = sorted(holding_df['年季'].unique())
    x_cols = ['MKT', 'SMB', 'HML', 'MOM']
    for q in tqdm(quarters, desc = "季度 alpha 回歸"):
        h_q =  holding_df[holding_df['年季'] == q]
        end_date = q.to_timestamp('M')
        start_date = (end_date - pd.DateOffset(months = window))
        f_win = factor_df[(factor_df['年月'] > start_date) & (factor_df['年月'] <= end_date)]
        s_win = stock_monthly_df[(stock_monthly_df['年月'] > start_date) & (stock_monthly_df['年月'] <= end_date)]
        for fund, h_f_q in h_q.groupby('基金代碼'):
            merged = pd.merge(h_f_q, s_win, on = 'key_code')
            if merged.empty:
                continue
            # 算出每個月的加權報酬
            merged['weighted_ret'] = merged['w'] * merged['ret_month']
            port_ret = merged.groupby('年月', as_index = False)['weighted_ret'].sum()
            port_ret.rename(columns = {'weighted_ret': 'ret_month'}, inplace = True)
            reg_data = pd.merge(port_ret, f_win, on = '年月')
            if len(reg_data) < 12:
                continue
            Y = reg_data['ret_month'] - reg_data['RF']
            X = sm.add_constant(reg_data[x_cols])
            try:
                model = sm.OLS(Y, X).fit()
                alpha_results.append({
                    '基金代碼': fund,
                    '年季': q,
                    'alpha': model.params['const']
                })
            except Exception as e:
                print(f"回歸失敗: 基金 {fund} 在季度 {q}，錯誤訊息: {e}")
                continue
    return pd.DataFrame(alpha_results)


def compute_active_share(holding_w: pd.DataFrame):
    '''
    計算 Active Share (同儕差異度) 當作基金 alpha 的 proxy
    計算個別基金的持倉與「全體主動基金平均持倉」的差異
    '''
    print("正在計算 Active Share (同儕差異度)...")
    # 建構 benchmark 持倉
    bench = holding_w.groupby(['年季', 'key_code'], as_index = False).agg(w = ('w', 'sum'))
    bench['total_w_season'] = bench.groupby('年季')['w'].transform('sum')
    bench['w_bench'] = bench['w'] / bench['total_w_season']
    bench = bench[['年季', 'key_code', 'w_bench']]
    # 計算 Active Share
    merged = holding_w.merge(bench, on = ['年季', 'key_code'], how = 'inner')
    merged['min_w'] = np.minimum(merged['w'], merged['w_bench'])
    merged_sum = merged.groupby(['基金代碼', '年季'], as_index = False).agg(min_w_sum = ('min_w', 'sum'))
    merged_sum['alpha'] = 1 - merged_sum['min_w_sum']
    return merged_sum[['基金代碼', '年季', 'alpha']]   


def compute_concentration(holding_w: pd.DataFrame):
    '''
    計算基金持倉的集中度, 當作基金 alpha 的 proxy
    計算每季持股檔數, 作為 TNA 的替代指標
    預期: 反向指標 (檔數越少 = 越集中 = 預期表現越好) -> 取 -Log(N)
    但台灣基金,觀察後是相反的
    '''
    print("正在計算 Concentration (集中度)")
    valid_h = holding_w[holding_w['w'] > 0.0001]
    conc = valid_h.groupby(['基金代碼', '年季'], as_index = False).agg(num_stocks = ('key_code', 'count'))
    conc['alpha'] = np.log(conc['num_stocks'])
    return conc[['基金代碼', '年季', 'alpha']]


def compute_hbm_skill(holding_df: pd.DataFrame, stock_monthly_df: pd.DataFrame, k):
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


def compute_carhart4_skill(fund_monthly_df: pd.DataFrame, factor_df: pd.DataFrame, window = WINDOW):
    '''計算基金的 Carhart 四因子 alpha'''
    print(f"正在計算基金的 Carhart 四因子 alpha (window = {window} 月)")
    fund_monthly_df = fund_monthly_df.sort_values(by = ['基金代碼', '年月'])
    results = []
    x_cols = ['MKT', 'SMB', 'HML', 'MOM']
    for fund, group in tqdm(fund_monthly_df.groupby('基金代碼'), desc = "基金 R2 回歸"):
        data = pd.merge(group, factor_df, on = '年月')
        if len(data) < window:
            continue
        qe_dates = data[data['年月'].dt.is_quarter_end]['年月']
        for qe in qe_dates:
            mask = data[data['年月'] <= qe]
            window_data = mask.tail(window)
            # 確保該計算視窗內的資料筆數足夠 (至少 80%)
            if len(window_data) < window * 0.8:
                continue
            Y = window_data['ret_month'] - window_data['RF']
            X = sm.add_constant(window_data[x_cols])
            try:
                model = sm.OLS(Y, X).fit()
                results.append({
                    '基金代碼': fund,
                    '年季': qe.to_period('Q'),
                    'alpha': model.params.const
                })              
            except Exception as e:
                continue
    return pd.DataFrame(results)


def compute_one_minus_r2_skill(fund_monthly_df: pd.DataFrame, factor_df: pd.DataFrame, window = WINDOW):
    '''計算持倉一減 R² 當作基金 alpha proxy'''
    print(f"正在計算基金 一減 R² 指標 (window = {window} 月)")
    fund_monthly_df = fund_monthly_df.sort_values(by = ['基金代碼', '年月'])
    results = []
    x_cols = ['MKT', 'SMB', 'HML', 'MOM']
    for fund, group in tqdm(fund_monthly_df.groupby('基金代碼'), desc = "基金 R2 回歸"):
        data = pd.merge(group, factor_df, on = '年月')
        if len(data) < window:
            continue
        qe_dates = data[data['年月'].dt.is_quarter_end]['年月']
        for qe in qe_dates:
            mask = data[data['年月'] <= qe]
            window_data = mask.tail(window)
            if len(window_data) < window * 0.8:
                continue
            Y = window_data['ret_month'] - window_data['RF']
            X = sm.add_constant(window_data[x_cols])
            try:
                model = sm.OLS(Y, X).fit()
                active_signal = 1 - model.rsquared
                results.append({
                    '基金代碼': fund,
                    '年季': qe.to_period('Q'),
                    'alpha': active_signal
                })              
            except Exception as e:
                continue
    return pd.DataFrame(results)


def compute_return_gap(fund_monthly_returns: pd.DataFrame, stock_quarterly_returns, fund_holdings: pd.DataFrame):
    '''計算基金的 return gap'''
    # 計算基金季報酬
    fund_q_ret = fund_monthly_returns.groupby(['基金代碼', pd.Grouper(key = '年月', freq = 'Q')]).agg(
        fund_q_ret_actual = ('ret_month', lambda s: (1 + s).prod() - 1)
        ).reset_index()
    fund_q_ret['年季'] = fund_q_ret['年月'].dt.to_period('Q')
    
    # 計算基準投資組合季報酬（用上一季持股）
    holding_begin_q = fund_holdings.copy()
    merged = pd.merge(holding_begin_q, stock_quarterly_returns, on = ['key_code', '年季'], how = 'inner')
    # equal weighting
    bh = merged.groupby(['基金代碼', '年季']).apply(
        lambda g: np.average(g['q_ret'], weights = g['w'])
    ).reset_index(name = 'bh_q_ret')
    
    gap = pd.merge(fund_q_ret, bh, on = ['基金代碼', '年季'], how = 'inner')
    gap['GAP'] = gap['fund_q_ret_actual'] - gap['bh_q_ret']
    gap = gap.rename(columns = {'GAP': 'alpha'})
    return gap[['基金代碼', '年季', 'alpha']]


def compute_tracking_error(fund_monthly_df: pd.DataFrame, factor_df: pd.DataFrame, window):
    '''
    用 Tracking Error 當作基金 alpha 的 proxy
    計算基金主動報酬的波動率，使用 Carhart MKT 當作大盤超額報酬
    '''
    print(f"正在計算基金 Tracking Error (風險主動性, window = {window})...")
    data = pd.merge(fund_monthly_df, factor_df[['年月', 'MKT', 'RF']], on = '年月', how = 'inner')
    data = data.sort_values(by = ['基金代碼', '年月'])
    data['fund_excess'] = data['ret_month'] - data['RF']
    data['active_resid'] = data['fund_excess'] - data['MKT']
    results = []
    for fund, group in tqdm(data.groupby('基金代碼'), desc = '計算 Tracking Error'):
        qe_dates = group[group['年月'].dt.is_quarter_end]['年月']
        for qe in qe_dates:
            mask = group['年月'] <= qe
            window_data = group[mask].tail(window)
            if len(window_data) < window * 0.8:
                continue
            te_val = window_data['active_resid'].std()
            results.append({
                '基金代碼': fund,
                '年季': qe.to_period('Q'),
                'alpha': te_val
            })
    return pd.DataFrame(results)


def compute_gtw_skill(holding_df: pd.DataFrame, stock_monthly_df: pd.DataFrame, k):
    '''計算持倉動能，是基於持倉權重變化與股票動能的乘積和，是衡量流量'''
    s_data = stock_monthly_df.sort_values(by = ['key_code', '年月']).copy()
    s_data['ret_plus_1'] = s_data['ret_month'] + 1
    s_data['mom6'] = s_data.groupby('key_code')['ret_plus_1'].transform(lambda x: x.rolling(k).apply(np.prod, raw = True) - 1)
    s_mom = s_data[s_data['年月'].dt.is_quarter_end][['key_code', '年月', 'mom6']]
    s_mom['年季'] = s_mom['年月'].dt.to_period('Q')
    h_curr = holding_df.copy()
    h_curr['prev_q'] = h_curr['年季'] - 1
    h_prev = h_curr[['基金代碼', '年季', 'key_code', 'w']].rename(columns = {'年季': 'prev_q', 'w': 'w_prev'})
    merged_w = pd.merge(h_curr, h_prev, on = ['基金代碼', 'prev_q', 'key_code'], how = 'outer')
    merged_w['w'] = merged_w['w'].fillna(0)
    merged_w['w_prev'] = merged_w['w_prev'].fillna(0)
    merged_w['w_gap'] = merged_w['w'] - merged_w['w_prev']
    final_df = pd.merge(merged_w, s_mom[['key_code', '年季', 'mom6']], on = ['key_code', '年季'], how = 'inner')
    gtw_skill = final_df.groupby(['基金代碼', '年季']).apply(lambda x: np.sum(x['w_gap'] * x['mom6'])).reset_index(name = 'alpha')
    return gtw_skill       

# 計算 GIA 函數
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

