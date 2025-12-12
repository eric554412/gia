# 模型與程式檔說明

檔案清單：
- [gia.py](gia.py)
- [return_gap_full_gia.py](return_gap_full_gia.py)
- [stock_quality_measure.py](stock_quality_measure.py)

**純模型說明（概念導向）**

- Carhart-alpha（共同基底）
  - 模型：以 Carhart 四因子做滾動回歸，擷取基金的超額報酬（alpha）。
  - 數學直覺：對每檔基金在一段時間窗口內估計
    $$(R_p - R_f) = \alpha + \beta_{MKT}MKT + \beta_{SMB}SMB + \beta_{HML}HML + \beta_{MOM}MOM + \varepsilon$$
  - 目的：alpha 被視為基金在已知因子之外的主動績效或資訊來源。

- Return Gap（在 `return_gap_full_gia.py` 中）
  - 概念：比較基金實際的季報酬與按其持股加權的股票基準回報，差異稱為 GAP：
    $$GAP = R_{fund}^{actual} - \sum_j w_{ij} R_{j}^{stock}\ (weighted\ benchmark)$$
  - 使用方式：以 $-GAP$ 當作基金的 alpha（視為反向策略或未被持股解釋的超額），再將其映射回股票。

- GIA（基金→股票反推，`gia.py` / `return_gap_full_gia.py`）
  - 目標：把基金層級的 alpha 向量 $S$ 透過基金×股票權重矩陣 $W$ 映射回股票層級分數 $g$（GIA）。
  - 理論表達：理想上滿足
    $$W g \approx S$$
    要求解 $g$，但 $W$ 通常非方陣且可能病態，因此使用奇異值分解（SVD）與截斷的廣義逆求解：
    $$g \approx W^{+}_{trunc} S$$
  - 直覺：GIA 表示哪些股票被高 alpha 的基金共同指認為貢獻 alpha 的標的。

- Stock Measure Quality（SMQ，`stock_quality_measure.py`）
  - 概念：對每季，對於每檔股票以持股權重把持有該股的基金 alpha 加權平均，得到該股的品質分數（SMQ）：
    $$SMQ_j = \frac{\sum_{i} w_{ij} \alpha_i}{\sum_i w_{ij}}\quad(通常為加權平均)$$
  - 直覺：若多數高 alpha 的基金持有該股，則該股 SMQ 較高。

- 參數搜尋與穩定化（Grid Search）
  - K-ratio / 截斷：在 SVD 解法中以 K（或 K ratio）保留主要基底，避開小奇異值導致的噪音放大。
  - Window：滾動回歸的時間窗口長短控制 alpha 的平滑與穩定性。
  - 評估指標：以 Spearman 相關（單調性）、違反次數（violations）、long-short 的 Newey‑West t 值等，挑選使分數具有預測力與穩健性的參數。

- 回測方法（共同）
  - 依股票分數將股票按季度分成十等分（decile），計算各投組的季報酬與 long‑short（top minus bottom）。
  - 使用 Newey‑West 調整的 t 值評估長短組績效顯著性；並用平均、波動、年化夏普與勝率評估穩定性。

- 概念性限制與注意事項
  - 反演假設：把基金 alpha 線性分配給持股是假設性的，若基金共同持股或持股稀疏，反演可能不穩定或含偏誤。
  - 訊號來源複雜：高 GIA/SMQ 可能來自共同持股、流動性、報表偏差，而非純粹 alpha；需搭配其他檢驗驗證。
  - 參數敏感：window 與 K ratio 會影響結果，故需做穩健性檢驗。

**（以下為程式相關參考，非模型說明）**
本檔案同時列出各程式的實作檔案與主要函數，供程式維護使用；若只需純模型描述，以上即可作為精簡說明。

**各檔案重點**:

**[gia.py](gia.py)**
- 目的：執行 Grid Search（Window 與 K ratio）以找到最佳化的 GIA 生成與回測設定。
- 主要函數：
  - `run_carhart_rolling`：對每檔基金在每個季度做滾動窗口（預設 18 月）Carhart 4-factor OLS，取季度末 alpha。
  - `compute_gia_grid_optimized`：對每季建立基金×股票權重矩陣 W，對 W 做一次 SVD，對所有 K ratio 使用 SVD 分解加速計算股票 GIA（針對不同 K 只需重用 SVD 結果）。
  - `backtest_single_decile`、`calc_monotonicity_score`、`build_slim_metrics_table`：回測、計算 Newey-West t 與單調性量表。
- 輸入：`fund_data/merged_fund_data.csv`、`fund_data/carhart_factor.csv`（UTF-16 LE）等。
- 輸出：在 console 輸出 Grid Search 表格與最佳組合、以及最終績效格式表；不會寫檔（可擴充）。
- 備註：使用 SVD 做加速；針對每季先計算一次 SVD，再依 k_ratio 快速產生結果。

**[return_gap_full_gia.py](return_gap_full_gia.py)**
- 目的：以「Return Gap」（基金實際季報酬減去按持股加權的股票基準）來反推 alpha，再用 truncated pseudo-inverse（SVD-based）計算 GIA，並做 K ratio 的 Grid Search。
- 主要函數：
  - `compute_return_gap`：計算基金季報酬實際值與由持股加權的基準之差（GAP），以 GAP 的負號作為 alpha（反向策略）。
  - `compute_gia`：使用截斷的廣義逆（SVD）根據 k_ratio 計算 stock-level alpha（GIA）。
  - 回測/評分函數與 `gia.py` 類似（decile、Spearman rho、Newey-West t）。
- 輸入：同樣需要基金月報、持股與股票資料；K_RATIO_LIST 在檔頭設定。
- 輸出：console 上顯示最佳 K 及最終績效表。

**[stock_quality_measure.py](stock_quality_measure.py)**
- 目的：計算 Stock Measure Quality (SMQ)，來源於 Cohen et al. (2005) 的思路：以基金 alpha 加權計算股票品質指標，然後回測檢驗 SMQ 對後續季報酬的預測力。
- 主要函數：
  - `run_carhart_rolling`、`prepare_fund_alpha`：同前，計算基金 alpha 與整理成季度級別。
  - `compute_stock_measure_quality`：對每季，將持股中各基金 alpha 以權重加權平均，產出 `SMQ`（股票分數）。支援 Winsor 去極值選項。
  - `trim_outliers_cross_section`：能在每月 cross-section 對股票月報酬做去極值（trim）以穩定季報酬計算。
  - `backtest_single_decile`、`portfolio_metrics_from_wide`：decile 回測與投組績效指標（平均、年化夏普、hit ratio 等）。
- 輸入：與其他檔案相同的資料集。
- 輸出：Console 列印 SMQ 的十等分回測表與投組績效指標。

**重要欄位/檔案依賴**:
- 因子檔：需要包含 MKT, SMB, HML, MOM, RF 等欄位（檔案為 `fund_data/carhart_factor.csv`）。
- 基金月報：`merged_fund_data.csv`，需有 `證券代碼`、`年月`、`單月ROI` 等欄位。
- 持股表：`fund_data/fund_data.csv`，需有 `證券代碼`、`年季`、`標的碼`、`投資比率％` 與（可選）`投資標的` 欄位。
- 股票月表：`fund_data/stock_return.csv`，需有 `證券代碼`、`年月`、`報酬率％_月`（或相似欄名）、價格欄位（若有價格過濾條件）。

**執行方式（範例）**:
- 直接執行單檔：
```bash
python gia.py
python return_gap_full_gia.py
python stock_quality_measure.py
```
- 注意：部分檔案預期的 CSV 編碼為 `UTF-16 LE` 且以 tab 分隔（例如因子與某些 stock_return），請依錯誤訊息檢查編碼與欄名。

**建議改進/注意事項**:
- 欄名與編碼敏感：若 CSV 欄位名稱或編碼不同，需調整函數內對應變數名或 read_csv 的 `encoding/sep`。
- 輸出與可重現：目前程式主要印到 console，可加上 `--out` 參數或將結果寫成 CSV/Excel 以利後續分析。
- 效能：`gia.py` 已使用 SVD 重用以加速不同 k_ratio，但如果基金/股票數量很大，建議採稀疏矩陣與分批處理、或使用專用線性代數庫以降低記憶體與計算時間。
- 測試：建議加入簡單的 unittest 或小型樣本資料集，以驗證各步驟（alpha 計算、GIA/SMQ 生成、回測）正確性。

---

如果你要，我可以：
- 把 `MODEL_SUMMARY.md` 改成更精簡或加入流程圖；
- 幫你把 console 輸出改成 CSV 檔案（例如 `best_params.csv`, `gia_results.csv`）；
- 或直接在這台機器跑任一腳本（需你確認 `fund_data/` 內資料是否完整且要我執行）。
