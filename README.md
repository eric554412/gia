# 因子特異性頻譜過濾下之廣義逆矩陣 Alpha

**台灣主動型股票基金之實證研究**

> 碩士論文｜研究生：胡益鳴　指導教授：羅秉政 博士
> 論文全文見 [`碩士論文＿胡益鳴.pdf`](碩士論文＿胡益鳴.pdf)，口試簡報見 [`論文簡報.pdf`](論文簡報.pdf)（LaTeX 原始檔於 [`beamer/`](beamer/)）。

## 研究概觀

本研究以 Wermers, Yao and Zhao (2012) 的**廣義逆矩陣 Alpha（Generalized Inverse Matrix Alpha, GIA）**模型為基礎，檢驗其在台灣股票市場的選股效力。GIA 的核心概念是：觀察到的是「基金層級」的績效與持股，但真正想知道的是「個股層級」的隱含 alpha；透過對基金持股矩陣做**截斷奇異值分解（truncated SVD）的偽逆**，可從眾多基金的共識中反解出個股 alpha，同時以頻譜過濾濾除尾端的隨機交易雜訊。

研究目的有三：

1. **驗證因子頻譜結構**：探討不同屬性的因子（共識型 vs. 私有型）在奇異值頻譜上是否呈現相異的物理結構。
2. **方法論創新**：打破過往文獻「固定截斷參數 K」的限制，提出**因子特異性參數校準**——不同因子使用各自最適的頻譜截斷比例，以優化訊號雜訊比。
3. **建構可獲利策略**：在資訊揭露遞延（約 60 天）下，檢驗 GIA 能否萃取具預測力的訊號，並在扣除交易成本與風險調整後創造顯著超額報酬。

實證結果（詳見論文）：GIA 贏家組的淨報酬（**17.95%**）顯著優於被動大盤 0050 ETF（**12.18%**）。

## 方法核心

令 `W` 為基金持股矩陣（M 檔基金 × N 檔個股）、`Ŝ` 為基金層級的 alpha 向量，個股層級的 GIA alpha 由截斷偽逆求得：

```
α_GIA = W⁺ Ŝ = V Σ_K⁺ Uᵀ Ŝ
```

其中 `Σ_K⁺` 只保留前 K 個奇異值（頻譜過濾）。本研究的創新在於 **K 依因子特性動態校準**，而非採用固定值。

## 專案結構

```
kendro_vincent/
├── model/                        # 核心程式
│   ├── factor_lib.py             # 共用函式庫：compute_gia()（截斷 SVD 偽逆）、資料前處理、Newey-West t 值
│   ├── data_bulider.py           # 主流程：逐因子算 fund alpha → 套 GIA → z-score → 產出 gia_score.csv
│   ├── fund_alpha.py             # 計算各項基金層級 alpha 指標 → 產出 fund_alpha_scores.csv
│   ├── gia.py                    # GIA 單因子完整示範流程
│   ├── return_gap_full_gia.py    # 以 Return Gap 為因子的 GIA 變體流程
│   │   # ── 基金經理人能力指標（各自作為 GIA 的輸入因子）──
│   ├── active_share.py           # Active Share（偏離同業共識程度）
│   ├── concentration.py          # 持股集中度 / 多元化
│   ├── tracking_error.py         # 追蹤誤差
│   ├── r_square.py               # 1 − R²（主動風險，未被四因子解釋的比例）
│   ├── holding_based_mom.py      # 持有型動能（持股是否為過去贏家）
│   ├── trading_mom.py            # 交易型動能（加碼方向是否對準贏家）
│   ├── holding_alpha.py          # 持股 Carhart 四因子 alpha
│   ├── stock_quality_measure.py  # 個股品質指標
│   │   # ── 回測與校準 ──
│   ├── backtest.py               # 個股依 GIA 分數分十等分回測、Newey-West t 值、繪圖
│   ├── backtest_fund.py          # 基金層級績效回測
│   ├── sim.py                    # K 參數掃描（0.01–1.00）校準最適截斷比例
│   └── test/                     # 探索性機器學習
│       ├── ridge_regression.py   #   Ridge（RidgeCV）以多因子預測次季個股報酬
│       └── xgb_regrssion.py      #   XGBoost 非線性預測
│
├── fund_data/                    # 資料層（原始輸入 + 產出）
│   ├── process_data.py           # 合併基金報酬與補漏資料的前處理
│   ├── fund_data.csv             # 原始：基金季持股（TEJ）
│   ├── stock_return.csv          # 原始：個股月報酬與收盤價（TEJ, UTF-16）
│   ├── carhart_factor.csv        # 原始：Carhart 四因子（MKT/SMB/HML/MOM/RF, UTF-16）
│   ├── fund_roi_monthly.csv      # 原始：基金月報酬（UTF-16）
│   ├── merged_fund_data*.csv     # 合併後基金資料
│   ├── 0050.csv                  # 0050 ETF 基準報酬
│   ├── gia_score.csv             # 產出：個股 GIA 分數
│   └── fund_alpha_scores.csv     # 產出：基金層級 alpha 分數
│
├── beamer/                       # 口試簡報（LaTeX Beamer 原始檔與圖）
├── build/                        # 簡報 / 文件建置產物
├── 論文簡報.pdf                   # 口試簡報（PDF，即 beamer 編譯結果）
└── 碩士論文＿胡益鳴.pdf            # 論文全文
```

## 資料

- **來源**：台灣經濟新報（TEJ）。部分原始檔為 UTF-16、Tab 分隔（`stock_return.csv`、`carhart_factor.csv`、`fund_roi_monthly.csv`）。
- **樣本**：台灣主動型國內股票基金；為消除倖存者偏誤，同時納入存續與已清算基金。
- **落後處理**：持股資料強制落後，以反映約 60 天的資訊揭露遞延。

## 執行流程

程式假設在**專案根目錄**下、以個股/基金資料位於 `fund_data/` 的相對路徑執行。概念上的順序為：

```
1. fund_data/process_data.py     # 合併基金資料（前處理）
2. model/fund_alpha.py           # 產出 fund_alpha_scores.csv（基金層級 alpha）
   model/data_bulider.py         # 產出 gia_score.csv（逐因子套 GIA + z-score）
3. model/sim.py                  # 掃描並校準各因子最適 K
4. model/backtest.py             # 分十等分回測、繪製累積報酬與績效表
```

## 相依套件

`pandas`、`numpy`、`scipy`、`statsmodels`（OLS / Newey-West HAC）、`matplotlib`、`tqdm`；機器學習的探索性模組另需 `scikit-learn`、`xgboost`。

## 注意事項

- 部分腳本（`model/active_share.py`、`model/concentration.py`、`model/gia.py`、`model/tracking_error.py`）目前含有指向本機的**絕對路徑**（讀取 `fund_roi_monthly.csv`），換機器執行前請改為相對路徑。
- `data_bulider.py`、`fund_alpha.py` 以 `import factor_lib as fl` 匯入共用函式庫，需從 `model/` 目錄執行或將其加入 `PYTHONPATH`。
