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
NW_LAGS = 4            
N_GROUPS = 10    # 10 分位數 (Decile)

# ==========================================
# 工具函數區塊
# ==========================================

def _newey_west_t(series, lags=6):
    y = pd.Series(series).dropna()
    if len(y) < 5:
        return np.nan, np.nan, len(y)
    res = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0]), len(y)

def calc_max_drawdown(return_series):
    wealth_index = (1 + return_series).cumprod()
    peaks = wealth_index.cummax()
    drawdowns = (wealth_index - peaks) / peaks
    return drawdowns.min()

def prep_fund_monthly_for_backtest(df_fund: pd.DataFrame, code_col='證券代碼', date_col='年月', ret_pct_col='單月ROI'):
    df = df_fund.copy()
    
    # 1. 自動清理所有欄位名稱的頭尾空白字元（防呆）
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. 檢查必備欄位是否存在
    for col in [code_col, date_col, ret_pct_col]:
        if col not in df.columns:
            raise KeyError(f"在資料中找不到 '{col}' 欄位！目前檔案內有的欄位為: {list(df.columns)}")
            
    # 3. 處理代碼對齊
    df['key_code'] = df[code_col].astype(str).str.strip() 
    
    # 4. 處理日期
    df['年月'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    
    # 5. 處理報酬率 (假設你的單月ROI是 % 數，例如 5.5 代表 5.5%，所以除以 100。若原本就是小數 0.055，請把 / 100 拿掉)
    df['ret_month'] = pd.to_numeric(df[ret_pct_col], errors='coerce') / 100
    
    return df[['key_code', '年月', 'ret_month']].dropna(subset=['ret_month'])

def prep_factor_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-16', sep='\t')
    except Exception:
        try:
             df = pd.read_csv(file_path, encoding='utf-8') 
        except Exception as e:
            print(f"[警告] 讀取因子資料失敗: {e}")
            return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    column_map = {'市場風險溢酬': 'MKT', '規模溢酬 (3因子)': 'SMB', '淨值市價比溢酬': 'HML', '動能因子': 'MOM', '年月': 'DATE'}
    
    rf_col_found = False
    for col in df.columns:
        if '無風險利率' in col:
            column_map[col] = 'RF'
            rf_col_found = True
            break
            
    df = df.rename(columns=column_map)
    if not rf_col_found and 'RF' not in df.columns:
        df['RF'] = 0.0

    req_factors = ['MKT', 'SMB', 'HML', 'MOM', 'RF']
    if 'DATE' not in df.columns:
        date_col = next((c for c in df.columns if '年月' in c or '日期' in c), None)
        if date_col: df = df.rename(columns={date_col: 'DATE'})
        else: return pd.DataFrame()
            
    df['年月'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m', errors='coerce')
    df['年季'] = df['年月'].dt.to_period('Q')
    
    for col in req_factors:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    q_factors = df.groupby('年季')[req_factors].agg(lambda x: (1 + x).prod() - 1).reset_index()
    return q_factors

def calc_all_alphas(wide_df, df_factors):
    results = {}
    if df_factors.empty: return results

    if not isinstance(wide_df.index, pd.PeriodIndex):
        wide_df.index = pd.PeriodIndex(wide_df.index, freq='Q')
    df_factors = df_factors.set_index('年季')
    if not isinstance(df_factors.index, pd.PeriodIndex):
        df_factors.index = pd.PeriodIndex(df_factors.index, freq='Q')
        
    merged = wide_df.join(df_factors, how='inner').dropna()
    if len(merged) < 12: return results
        
    X = merged[['MKT', 'SMB', 'HML', 'MOM']]
    X = sm.add_constant(X)
    
    for col in wide_df.columns:
        if col not in merged.columns: continue
        rf_correction = merged['RF'] if 'RF' in merged.columns else 0
        Y = merged[col] if str(col) == 'long_short' else merged[col] - rf_correction
        
        try:
            model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            results[col] = (model.params['const'], model.tvalues['const'], model.params['MKT'])
        except Exception:
            results[col] = (np.nan, np.nan, np.nan)
    return results

def monthly_to_quarterly_return(df_monthly: pd.DataFrame, key_col='key_code', ret_col='ret_month'):
    d = df_monthly.copy()
    d['年季'] = d['年月'].dt.to_period('Q')
    qret = d.groupby([key_col, '年季'], as_index=False).agg(
        q_ret=(ret_col, lambda x: (1 + x).prod() - 1),
        n_months=(ret_col, 'count')
    )
    return qret[qret['n_months'] == 3].reset_index(drop=True)

def backtest_single_factor(score_df, qret_df, score_col, n_groups=5, nw_lags=NW_LAGS):
    g = score_df.dropna(subset=[score_col]).copy() # 確保過濾掉該因子為空的基金
    if not isinstance(g['年季'].dtype, pd.PeriodDtype):
         g['年季'] = pd.PeriodIndex(g['年季'], freq='Q')

    def assign_groups(dfq):
        try:
            dfq['group'] = pd.qcut(dfq[score_col].rank(method='first'), q=n_groups, labels=False, duplicates='drop') + 1
        except Exception:
             dfq['group'] = np.nan
        return dfq

    g_grp = g.groupby('年季', group_keys=False).apply(assign_groups).reset_index(drop=True)
    g_grp = g_grp.rename(columns={'年季': 'formation_q'})
    g_grp['持有季'] = g_grp['formation_q'] + 1
    
    qret = qret_df.copy()
    if not isinstance(qret['年季'].dtype, pd.PeriodDtype):
        qret['年季'] = pd.PeriodIndex(qret['年季'], freq='Q')

    merged = pd.merge(g_grp, qret, left_on=['key_code', '持有季'], right_on=['key_code', '年季'], how='left')
    valid_merged = merged.dropna(subset=['q_ret'])

    avg_total_funds = valid_merged.groupby('formation_q')['key_code'].nunique().mean()

    port = valid_merged.groupby(['formation_q', 'group'], as_index=False).agg(ret_mean=('q_ret', 'mean'), n_stocks=('key_code', 'count'))
    wide = port.pivot(index='formation_q', columns='group', values='ret_mean').sort_index()
    avg_stocks_per_group = port.groupby('group')['n_stocks'].mean()
    
    stock_sets = valid_merged.groupby(['formation_q', 'group'])['key_code'].apply(set).unstack('group')
    turnover_df = pd.DataFrame(index=stock_sets.index, columns=stock_sets.columns, dtype=float)
    
    for group in stock_sets.columns:
        prev_set = set()
        for q in stock_sets.index:
            curr_set = stock_sets.at[q, group]
            if isinstance(curr_set, set) and len(curr_set) > 0:
                if len(prev_set) > 0: turnover_df.at[q, group] = len(curr_set - prev_set) / len(curr_set)
            prev_set = curr_set if isinstance(curr_set, set) else set()
            
    avg_turnover_per_group = turnover_df.mean()

    for k in range(1, n_groups + 1):
        if k not in wide.columns: wide[k] = np.nan
    wide = wide[sorted(wide.columns)]
    
    if n_groups in wide.columns and 1 in wide.columns:
        wide['long_short'] = wide[n_groups] - wide[1]
    else: wide['long_short'] = np.nan

    rows = []
    cols = list(range(1, n_groups + 1)) + ['long_short']
    for col in cols:
        if col in wide.columns:
            m, t, _ = _newey_west_t(wide[col], lags=nw_lags)
            rows.append({
                'portfolio': col, 
                'mean': m, 't': t, 
                'avg_stocks': avg_stocks_per_group.get(col, np.nan) if isinstance(col, int) else np.nan,
                'turnover': avg_turnover_per_group.get(col, np.nan) if isinstance(col, int) else np.nan
            })
            
    return wide, pd.DataFrame(rows).set_index('portfolio'), avg_total_funds

def build_slim_metrics_table(wide, summary_raw, alpha_results, periods_per_years=4):
    cols = [*sorted(c for c in wide.columns if isinstance(c, int)), 'long_short']
        
    out = []
    for col in cols:
        r_clean = wide[col].dropna() if col in wide.columns else pd.Series(dtype=float)
        if r_clean.empty:
            mean_q = std_q = mtv = sharpe_ann = total_ret = mdd = vol_ann = np.nan
        else:
            mean_q, std_q = r_clean.mean(), r_clean.std()
            vol_ann = std_q * np.sqrt(periods_per_years)
            sharpe_ann = np.nan if (pd.isna(std_q) or std_q == 0) else (mean_q / std_q) * np.sqrt(periods_per_years)
            total_ret, mdd = (1 + r_clean).prod() - 1, calc_max_drawdown(r_clean)      
            
        tval = summary_raw.loc[col, 't'] if col in summary_raw.index else np.nan
        avg_stocks = summary_raw.loc[col, 'avg_stocks'] if col in summary_raw.index else np.nan
        turnover = summary_raw.loc[col, 'turnover'] if col in summary_raw.index else np.nan
        alpha_val, alpha_t, beta_val = alpha_results.get(col, (np.nan, np.nan, np.nan))
        
        out.append([col, mean_q * 100, std_q * 100, vol_ann * 100, sharpe_ann, tval, total_ret * 100, mdd * 100, avg_stocks, turnover, alpha_val * 100, alpha_t, beta_val])
    
    slim_num = pd.DataFrame(out, columns=['portfolio', 'mean_pct', 'std_pct', 'vol_ann_pct', 'sharpe_annual', 't值', 'total_ret_pct', 'mdd_pct', 'avg_stocks', 'turnover', 'alpha_pct', 'alpha_t', 'beta']).set_index('portfolio')
    
    return pd.DataFrame({
        '季均檔數': slim_num['avg_stocks'].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}"), 
        '季均換股率': slim_num['turnover'].apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%"),
        '年化報酬': (slim_num['mean_pct'] * periods_per_years / 100).apply(lambda x: f"{x:.2%}" if not pd.isna(x) else ""),
        '季平均報酬': slim_num['mean_pct'].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}%"), 
        '年化波動率': slim_num['vol_ann_pct'].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}%"),
        '夏普比率': slim_num['sharpe_annual'].apply(lambda x: "" if pd.isna(x) else f"{x:.3f}"),
        't值': slim_num['t值'].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}"),
        '總累積報酬': slim_num['total_ret_pct'].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}%"),
        '最大回撤': slim_num['mdd_pct'].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}%"),
        "Carhart Alpha": slim_num['alpha_pct'].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}%"),
        'Alpha t值': slim_num['alpha_t'].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}"),
        'Market Beta': slim_num['beta'].apply(lambda x: "" if pd.isna(x) else f"{x:.3f}") 
    })

def calc_spearman_monotonicity(wide_df, n_groups):
    cols = list(range(1, n_groups + 1))
    valid_cols = [c for c in cols if c in wide_df.columns]
    if len(valid_cols) < 3: return np.nan, np.nan
    return spearmanr(np.array(valid_cols), wide_df[valid_cols].mean().values)

def plot_performance(wide, n_groups, factor_name):
    if wide.empty: return
    cum_ret = (1 + wide).cumprod()
    if not isinstance(cum_ret.index, pd.DatetimeIndex):
        cum_ret.index = cum_ret.index.to_timestamp()
    
    plt.figure(figsize=(10, 5))
    if 'long_short' in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret['long_short'], label='Long-Short (Q5 - Q1)', color='#d62728', linewidth=2.5, zorder=10)
    if n_groups in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret[n_groups], label=f'Q5 (Top)', color='blue', linestyle='-', linewidth=1.5, alpha=0.8)
    if 1 in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret[1], label='Q1 (Bottom)', color='green', linestyle='-', linewidth=1.5, alpha=0.8)

    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    plt.title(f'基金 Alpha 因子策略累積淨值: {factor_name}', fontsize=14, fontweight='bold')
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('累積淨值', fontsize=12)
    plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程式區塊
# ==========================================
def run_multiple_factor_backtests(df_scores_input):
    print("===== 開始執行 9 個 Alpha 因子的逐一回測 =====")
    
    try:
        df_fund_raw = pd.read_csv("fund_data/fund_roi_monthly.csv", encoding='utf-16', sep='\t')
        df_factors = prep_factor_data("fund_data/carhart_factor.csv")
    except Exception as e:
        print(f"錯誤：無法讀取資料 - {e}")
        return

    fund_m = prep_fund_monthly_for_backtest(df_fund_raw)
    fund_q = monthly_to_quarterly_return(fund_m)

    df_scores = df_scores_input.copy()
    if '基金代碼' in df_scores.columns:
        df_scores = df_scores.rename(columns={'基金代碼': 'key_code'})
    df_scores['key_code'] = df_scores['key_code'].astype(str).str.strip()
    
    alpha_cols = [c for c in df_scores.columns if c.startswith('alpha_')]
    if not alpha_cols:
        print("錯誤: 找不到任何 alpha_ 開頭的有效欄位，停止回測。")
        return

    # 儲存每個因子的單調性結果
    monotonicity_summary = []

    for factor in alpha_cols:
        factor_name = factor.replace('alpha_', '')
        print("\n" + "="*80)
        print(f"正在分析因子: 【{factor_name}】")
        print("="*80)
        
        wide, summary, avg_total_funds = backtest_single_factor(
            df_scores, 
            fund_q, 
            score_col=factor, 
            n_groups=N_GROUPS,
            nw_lags=NW_LAGS
        )

        if wide.empty:
            print(f"【{factor_name}】無法產生有效回測結果，跳過。")
            continue

        alpha_results = calc_all_alphas(wide, df_factors)
        slim_fmt = build_slim_metrics_table(wide, summary, alpha_results)
        rho, p_val = calc_spearman_monotonicity(wide, N_GROUPS)

        print(f"[市場概況] 該因子平均每季涵蓋基金檔數: {avg_total_funds:.0f} 檔")
        print(f"[單調性檢定] Spearman Rho: {rho:.4f} (p-value: {p_val:.4f})")
        print("-" * 80)
        print(slim_fmt)
        
        # 繪圖
        plot_performance(wide, N_GROUPS, factor_name)
        
        # 紀錄結果
        monotonicity_summary.append({
            'Factor': factor_name,
            'Spearman_Rho': rho,
            'P_Value': p_val,
            'Q5_Q1_Mean_Ret': (wide['long_short'].mean() * 100) if 'long_short' in wide.columns else np.nan
        })

    # === 印出總結表 ===
    print("\n" + "="*80)
    print("=== 【全部 9 個因子的單調性總結】 ===")
    print("="*80)
    df_summary = pd.DataFrame(monotonicity_summary).sort_values(by='Spearman_Rho', ascending=False).reset_index(drop=True)
    
    # 格式化輸出
    df_summary['Spearman_Rho'] = df_summary['Spearman_Rho'].apply(lambda x: f"{x:.4f}")
    df_summary['P_Value'] = df_summary['P_Value'].apply(lambda x: f"{x:.4f}")
    df_summary['Q5_Q1_Mean_Ret'] = df_summary['Q5_Q1_Mean_Ret'].apply(lambda x: f"{x:.2f}%")
    
    print(df_summary.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    file_path = "fund_data/fund_alpha_scores.csv"
    if os.path.exists(file_path):
        df_my_scores = pd.read_csv(file_path, encoding="utf-8") 
        run_multiple_factor_backtests(df_my_scores)
    else:
        print(f"找不到 {file_path}，請確認路徑與前置檔案是否已生成。")