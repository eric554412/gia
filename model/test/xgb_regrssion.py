import pandas as pd
import numpy as np
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import re

# 固定參數
FIXED_XGB_PARAMS = {
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': 1,
    'random_state': 42,
    'verbosity': 0
}
# 可調參數範圍
XGB_PARAM_GRID = {
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'reg_alpha': np.arange(0.2, 1.1, 0.2),
}

MIN_PRICE = 10
NW_LAGS = 4
CV_FOLDS = 5


def to_month_end(s):
    '''將各種日期格式轉換為月末日期'''
    s_str = s.astype(str)
    d1 = pd.to_datetime(s_str, format = '%Y%m', errors = 'coerce')
    d2 = pd.to_datetime(s_str, format = '%Y-%m', errors = 'coerce')
    d3 = pd.to_datetime(s_str, errors = 'coerce')
    d = d1.fillna(d2).fillna(d3)
    return d.dt.to_period('M').dt.to_timestamp('M')


def extract_code_token(s):
    '''把證券代碼字串標準化，取出主要代碼部分'''
    s = str(s).strip()
    if not s:
        return ""
    tok = s.split()[0].split('-')[0]
    return re.sub(r'[^0-9A-Z]', '', tok)


def prep_holding_from_fund_data(fund_data: pd.DataFrame, fund_col = '證券代碼', q_col = '年季', 
                                stock_id = '標的碼', weight_col = '投資比率％'):
    ''''從基金資料中整理出持股資料，並將持股權重標準化'''
    h = fund_data.copy()
    h['基金代碼'] = h[fund_col].astype(str).str.strip()
    h['年季'] = pd.PeriodIndex(h[q_col].astype(str).str.strip(), freq = 'Q')
    h['key_code'] = h[stock_id].astype(str).apply(extract_code_token)
    h['w_raw'] = pd.to_numeric(h[weight_col], errors = 'coerce') / 100
    h = h.groupby(['基金代碼', '年季', 'key_code'], as_index = False).agg(w_raw = ('w_raw', 'sum'))
    h= h[h['w_raw'] > 0].copy()
    h['w'] = h['w_raw'] / h.groupby(['基金代碼', '年季'])['w_raw'].transform('sum')
    return h[['基金代碼', '年季', 'key_code', 'w']]


def prep_stock_monthly_for_backtest(df_stock: pd.DataFrame, code_col = '證券代碼', date_col = '年月',
                                    ret_pct_col = '報酬率％_月', price_col = '收盤價(元)_月'):
    '''從股票月資料中整理出回測所需的欄位'''
    df = df_stock.copy()
    part = df[code_col].astype(str).str.strip().str.split(r'\s+', n = 1, expand = True)
    df['key_code'] = part[0].apply(extract_code_token)
    # 轉換年月為月末日期
    df['年月'] = pd.to_datetime(df[date_col].astype(str), format = '%Y%m', errors = 'coerce').dt.to_period('M').dt.to_timestamp('M') 
    df['ret_month'] = pd.to_numeric(df[ret_pct_col], errors = 'coerce') / 100
    df['price_month_end'] = pd.to_numeric(df.get(price_col, np.nan), errors = 'coerce')
    return df[['key_code', '年月', 'ret_month', 'price_month_end']]


def monthly_to_quarterly_return(df_monthly: pd.DataFrame):
    '''將月報酬率轉換為季報酬率'''
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q')
    qret = d.groupby(['key_code', '年季'], as_index = False).agg(
        q_ret = ('ret_month', lambda s: (1 + s).prod() - 1),
        n_months = ('ret_month', 'count')
    )
    return qret[qret['n_months'] == 3].reset_index(drop = True)


def build_entry_eligibility(stock_m: pd.DataFrame, min_price):
    '''季末篩選股票, 價格需高於 min_price 才有資格進入下一季投資組合'''
    sm = stock_m.copy()
    sm['年季'] = sm['年月'].dt.to_period('Q')
    sm['is_qe'] = sm['年月'].dt.is_quarter_end
    qe = sm[sm['is_qe']].copy()
    qe['持有季'] = (qe['年季'] + 1).astype('period[Q]')
    qe['eligible'] = (qe['price_month_end'] >= min_price).astype(int)
    return qe[['key_code', '持有季', 'eligible']]


def run_ml_stock_prediction_xgboost_cv(holding_w: pd.DataFrame, stock_q: pd.DataFrame, param_grid,
                                       fixed_params, min_stocks = 10, lookback_quarters = 2,
                                       ):
    '''使用 XGBoost 回歸模型進行股票報酬率預測，並進行交叉驗證調參'''
    hw = holding_w.copy()
    sq = stock_q.copy()
    hw['年季'] = pd.PeriodIndex(hw['年季'], freq = 'Q')
    sq['年季'] = pd.PeriodIndex(sq['年季'], freq = 'Q')
    
    sq = sq.sort_values(['key_code', '年季'])
    sq['target_next_ret'] = sq.groupby('key_code')['q_ret'].shift(-1)
    # 把 T 季持股權重與 T+1 季報酬合併在一起
    merged = pd.merge(hw, sq[['key_code', '年季', 'target_next_ret']], 
                      on = ['key_code', '年季'], how = 'inner').dropna(subset = ['target_next_ret'])
    
    unique_quarters = sorted(merged['年季'].unique())
    start_idx = lookback_quarters
    quarter_data = {}
    # 準備每季的訓練資料
    for q in unique_quarters:
        df_q = merged[merged['年季'] == q]
        # X 為這季的持股矩陣
        X_matrix = df_q.pivot(index = 'key_code', columns = '基金代碼', values = 'w').fillna(0)
        # Y 為下一季的報酬
        y_series = df_q[['key_code', 'target_next_ret']].drop_duplicates().set_index('key_code')['target_next_ret']
        common = X_matrix.index.intersection(y_series.index)
        if len(common) >= min_stocks:
            quarter_data[q] = (X_matrix.loc[common], y_series.loc[common])
    final_results = []
    n_stocks_list = []
    # 滾動訓練與預測
    for i in tqdm(range(start_idx, len(unique_quarters)), desc = '滾動訓練'):
        target_q = unique_quarters[i]
        train_qs = unique_quarters[i - lookback_quarters:i]
        X_train_list = []
        Y_train_list = []
        for tq in train_qs:
            d = quarter_data.get(tq)
            if d:
                X_train_list.append(d[0])
                Y_train_list.append(d[1])
        target_data = quarter_data.get(target_q)
        if not X_train_list or target_data is None:
            continue
        X_train = pd.concat(X_train_list).fillna(0)
        Y_train = pd.concat(Y_train_list)
        X_target, _ = target_data
        # 確保訓練與預測的欄位一致
        X_target = X_target.reindex(columns = X_train.columns, fill_value = 0)
        try:
            gs = GridSearchCV(
                XGBRegressor(**fixed_params),
                param_grid, 
                cv = CV_FOLDS,
                scoring = 'neg_mean_squared_error',
                n_jobs = -1
            )
            gs.fit(X_train, Y_train)
            y_pred = gs.best_estimator_.predict(X_target)
            n_stocks_list.append(len(X_target))
            final_results.append(pd.DataFrame({
                '年季': target_q,
                'key_code': X_target.index,
                'predicted_ret': y_pred
            }))
        except:
            continue
    return (
        pd.concat(final_results, ignore_index = True) if final_results else pd.DataFrame(),
        float(np.mean(n_stocks_list)) if n_stocks_list else 0.0
        )


def _newey_west_t(series, lags = NW_LAGS):
    '''計算 Newey-West 調整後的 t 統計量'''
    y = pd.Series(series).dropna()
    if len(y) < 5:
        return np.nan, np.nan, len(y)
    res = sm.OLS(y.values, np.ones((len(y), 1))).fit(cov_type = 'HAC', cov_kwds = {'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0]), len(y)


def backtest_single_decile(score_df: pd.DataFrame, qret_df: pd.DataFrame, eligibility_df: pd.DataFrame,
                           score_col = 'predicted_ret', n_groups = 10):
    '''對股票分數進行十分位數分組後回測'''
    sd = score_df.copy()
    sd['年季'] = pd.PeriodIndex(sd['年季'], freq = 'Q')
    # 如果同分出現，則按照出現順序排序
    sd['group'] = sd.groupby('年季')[score_col].transform(
        lambda x: pd.qcut(x.rank(method = 'first'), n_groups, labels = False) + 1
    )
    sd['持有季'] = sd['年季'] + 1
    if eligibility_df is not None:
        elig = eligibility_df.copy()
        elig['持有季'] = pd.PeriodIndex(elig['持有季'], freq = 'Q')
        sd = sd.merge(elig, on = ['key_code', '持有季'], how = 'left')
        sd = sd[sd['eligible'].fillna(0).eq(1)]
    merged = pd.merge(
        sd, qret_df, 
        left_on = ['key_code', '持有季'], 
        right_on = ['key_code', '年季']
    )
    port = merged.groupby(['持有季', 'group'])['q_ret'].mean().unstack(level = 'group')
    port['long_short'] = port[n_groups] - port[1]
    rows = []
    for col in list(range(1, n_groups + 1)) + ['long_short']:
        m, t, _ = _newey_west_t(port[col], lags = NW_LAGS)
        rows.append({'portfolio': col, 'mean_ret': m, 'nw_t': t})
    return port, pd.DataFrame(rows).set_index('portfolio')


def build_slim_metrics_table(wide: pd.DataFrame, summary_raw):
    '''建立績效表'''
    out = []
    for col in wide.columns:
        r = wide[col].dropna()
        mean_q = r.mean()
        std_q = r.std()
        sharpe_q = (mean_q / std_q) * 2 if std_q != 0 else np.nan
        tval = summary_raw.loc[col, 'nw_t'] if col in summary_raw.index else np.nan
        out.append([col, mean_q * 100, std_q * 100, sharpe_q, tval])
    res_df = pd.DataFrame(out, columns = ['portfolio', 'mean_ret_pct', 'std_pct', 'sharpe_ann', 'nw_t']).set_index('portfolio')
    f_p = lambda x: f"{x:.2f}%"
    f_v = lambda x: f"{x:.3f}"
    f_t = lambda x: f"{x:.2f}"
    return pd.DataFrame({
        'mean_pct': res_df['mean_ret_pct'].apply(f_p),
        'std_pct': res_df['std_pct'].apply(f_p),
        'sharpe_ann': res_df['sharpe_ann'].apply(f_v),
        't值': res_df['nw_t'].apply(f_t)
    })

def main():
    print("=== 啟動最佳化回測 (XGBoost) ===")
    try:
        df_fund = pd.read_csv('fund_data/merged_fund_data.csv')
        df_holding = pd.read_csv('fund_data/fund_data.csv')
        df_stock = pd.read_csv('fund_data/stock_return.csv', encoding = 'utf-16', sep = '\t')
    except Exception as e:
        print("錯誤: 找不到資料，請確認路徑")
        return       
    holding_data = prep_holding_from_fund_data(df_holding) 
    stock_m = prep_stock_monthly_for_backtest(df_stock)
    stock_q = monthly_to_quarterly_return(stock_m)
    entry_elig = build_entry_eligibility(stock_m, MIN_PRICE)
    
    pred_df, avg_n = run_ml_stock_prediction_xgboost_cv(
        holding_w=holding_data,
        stock_q=stock_q,
        param_grid=XGB_PARAM_GRID,
        fixed_params=FIXED_XGB_PARAMS
    )
    
    if not pred_df.empty:
        wide, summary = backtest_single_decile(pred_df, stock_q, entry_elig, score_col = 'predicted_ret')
        final_report = build_slim_metrics_table(wide, summary)
        print("\n" + "="*50)
        print(f"平均每季樣本數 (N): {int(avg_n)}")
        print("="*50)
        print(f"=== 最終績效報表 (Walk-Forward) ===")
        print(final_report)
        print("="*50)
    else:
        print("警告：回測結果為空，請檢查資料日期範圍。")


if __name__ == '__main__':
    main()
        
    
    