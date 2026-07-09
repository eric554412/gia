import pandas as pd
import numpy as np
import factor_lib as fl
from tqdm import tqdm

WINDOW = 36
TE_WINDOW = 12

def main():
    print("===== 開始計算基金層級 Alpha Proxy (包含9個因子，僅保留有效持股母體) =====")
    try:
        print("讀取檔案...")
        df_holding = pd.read_csv("fund_data/fund_data.csv", encoding="utf-8")
        df_factor = pd.read_csv("fund_data/carhart_factor.csv", encoding="utf-16", sep='\t')
        df_stock = pd.read_csv("fund_data/stock_return.csv", encoding="utf-16", sep='\t')
        df_fund = pd.read_csv("fund_data/fund_roi_monthly.csv", encoding="utf-16", sep='\t')
    except Exception as e:
        print(f"讀取檔案失敗: {e}")
        return
    
    print("資料前處理...")
    factor_data = fl.prepare_factor_data(df_factor)
    stock_m = fl.prep_stock_monthly_for_backtest(df_stock)
    stock_q = fl.monthly_to_quarterly_return(stock_m)
    holding_raw = fl.prep_holding_from_fund_data(df_holding) 
    holding = holding_raw.copy()
    holding['年季'] = holding['年季'] + 1  # 在 2023Q2 已知的持股資訊是 2023Q1 的持股
    
    # 【關鍵過濾】抓出有持股明細的基金名單，並以此過濾月報酬資料
    # 這樣 Carhart4Alpha 等指標就不會去浪費時間算無效基金了
    valid_funds = holding['基金代碼'].unique()
    fund_m = fl.prep_fund_data(df_fund)
    fund_m = fund_m[fund_m['基金代碼'].isin(valid_funds)].copy()
    
    # 包含你要求的 9 個因子 (加入 Carhart4Alpha)
    factor_list = [
        ("Active_share", lambda: fl.compute_active_share(holding)),
        ("Concentration", lambda: fl.compute_concentration(holding)),
        ("HoldingAlpha", lambda: fl.compute_holding_alpha_skill(holding, stock_m, factor_data, window=WINDOW)),
        ("Carhart4Alpha", lambda: fl.compute_carhart4_skill(fund_m, factor_data, window=WINDOW)), # 第9個因子加在這裡
        ("HBM", lambda: fl.compute_hbm_skill(holding, stock_m, k=6)),
        ("OneMinusR2", lambda: fl.compute_one_minus_r2_skill(fund_m, factor_data, window=WINDOW)),
        ("ReturnGap", lambda: fl.compute_return_gap(fund_m, stock_q, holding)),
        ("TrackingError", lambda: fl.compute_tracking_error(fund_m, factor_data, window=TE_WINDOW)),
        ("TradingMoM", lambda: fl.compute_gtw_skill(holding, stock_m, k=6))
    ]
    
    # 建立一個基礎 DataFrame，只包含實際「有持股」的 (基金代碼, 年季) 組合
    all_fund_features = holding[['基金代碼', '年季']].drop_duplicates().reset_index(drop=True)
    
    for name, func_wrapper in tqdm(factor_list, desc="計算基金因子"):
        try:
            fund_score = func_wrapper()
        except Exception as e:
            print(f"計算 {name} 失敗: {e}")
            continue
            
        if fund_score is None or fund_score.empty:
            print(f"{name} 計算結果為空, 跳過")
            continue
            
        fund_score = fund_score[['基金代碼', '年季', 'alpha']].copy()
        fund_score = fund_score.rename(columns={'alpha': f'alpha_{name}'})
        
        # 使用 Left Join，強制以有持股的母體為基準，把算出來的分數貼上去
        all_fund_features = pd.merge(
            all_fund_features,
            fund_score,
            on=['基金代碼', '年季'],
            how='left'
        )
    
    print("處理缺失值與儲存...")
    feat_cols = [col for col in all_fund_features.columns if col.startswith('alpha_')] 
    
    # 剔除全部因子都是空值的列
    all_fund_features = all_fund_features.dropna(subset=feat_cols, how='all')
    all_fund_features = all_fund_features.sort_values(['基金代碼', '年季'])
    
    # 儲存
    out_path = "fund_data/fund_alpha_scores.csv"
    all_fund_features.to_csv(out_path, index=False, encoding="utf-8")
    print(f"完成！共輸出 9 個 Alpha 因子，已過濾無持股基金。儲存至 {out_path}")

if __name__ == "__main__":
    main()