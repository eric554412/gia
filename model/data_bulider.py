import pandas as pd
import numpy as np
import factor_lib as fl
from tqdm import tqdm

WINDOW = 36
TE_WINDOW = 12


def main():
    print("===== 開始建構GIA SCORE =====")
    try:
        print("讀取檔案...")
        df_holding = pd.read_csv("fund_data/fund_data.csv", encoding = "utf-8" )
        df_factor = pd.read_csv("fund_data/carhart_factor.csv", encoding = "utf-16", sep = '\t')
        df_stock = pd.read_csv("fund_data/stock_return.csv", encoding = "utf-16", sep = '\t')
        df_fund = pd.read_csv("fund_data/fund_roi_monthly.csv", encoding = "utf-16", sep = '\t')
    except Exception as e:
        print(f"讀取檔案失敗: {e}")
        return
    
    print("資料前處理")
    factor_data = fl.prepare_factor_data(df_factor)
    stock_m = fl.prep_stock_monthly_for_backtest(df_stock)
    stock_q = fl.monthly_to_quarterly_return(stock_m)
    holding_raw = fl.prep_holding_from_fund_data(df_holding) 
    holding = holding_raw.copy()
    holding['年季'] = holding['年季'] + 1  # 在 2023Q2 已知的持股資訊是 2023Q1 的持股
    fund_m = fl.prep_fund_data(df_fund)
    # 格式: (因子名稱, 執行函數, 最佳 k_ratio)
    factor_list = [
        (
            "Active_share",
            lambda: fl.compute_active_share(holding),
            0.5
        ),
        (
            "Concentration",
            lambda: fl.compute_concentration(holding),
            0.4
        ),
        (
            "HoldingAlpha",
            lambda: fl.compute_holding_alpha_skill(holding, stock_m, factor_data, window = WINDOW),
            0.5
        ),        
        (
            "HBM",
            lambda: fl.compute_hbm_skill(holding, stock_m, k = 6),
            0.95
        ),
        (
            "OneMinusR2",
            lambda: fl.compute_one_minus_r2_skill(fund_m, factor_data, window = WINDOW),
            0.5
        ),
        (
            "ReturnGap",
            lambda: fl.compute_return_gap(fund_m, stock_q, holding),
            0.3
        ),
        (
            "TrackingError",
            lambda: fl.compute_tracking_error(fund_m, factor_data, window = TE_WINDOW),
            0.6
        ),
        (
            "TradingMoM",
            lambda: fl.compute_gtw_skill(holding, stock_m, k = 6),
            0.65
        )
    ]
    all_features = pd.DataFrame()
    for name, func_wrapper, best_k in tqdm(factor_list, desc = "計算因子"):
        # 1. 計算基金分數
        try:
            fund_score = func_wrapper()
        except Exception as e:
            print(f"計算 {name} 失敗: {e}")
            continue
        if fund_score.empty:
            print(f"{name} 計算結果為空, 跳過")
            continue
        # 2. 執行 GIA 反向解出股票 alpha
        stock_score = fl.compute_gia(fund_score, holding, best_k)
        if stock_score.empty:
            print(f"{name} GIA 計算結果為空, 跳過")
            continue
        stock_score['z_score'] = stock_score.groupby('年季')['GIA'].transform(
            lambda x: (x - x.mean()) / x.std() 
        )
        stock_score = stock_score.rename(columns = {'z_score': f'feat_{name}'})
        stock_score = stock_score[['年季', 'key_code', f'feat_{name}']]
        # 4. 合併所有特徵
        if all_features.empty:
            all_features = stock_score
        else:
            all_features = pd.merge(
                all_features,
                stock_score,
                on = ['年季', 'key_code'],
                how = 'outer'
            )
    
    print("處理缺失值與儲存...")
    feat_cols = [col for col in all_features.columns if col.startswith('feat_')] 
    all_features = all_features.dropna(subset = feat_cols, how = 'all')
    all_features = all_features.sort_values(['年季', 'key_code'])
    # 儲存
    out_path = "fund_data/gia_score.csv"
    all_features.to_csv(out_path, index = False, encoding = "utf-8")


if __name__ == "__main__":
    main()