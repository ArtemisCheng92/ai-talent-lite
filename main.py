from __future__ import annotations

from typing import Dict, List, Any, Tuple
import io
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==== Lazy imports (加速 Render 冷啟動) ====
# 只有在真的需要跑模型/分群時才載入 numpy/pandas/sklearn，
# 讓首頁與靜態頁面幾乎秒開。

@lru_cache(maxsize=1)
def _lazy_import_ml() -> None:
    global np, pd
    global ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder, SimpleImputer
    global RandomForestRegressor, ElasticNetCV, train_test_split, r2_score, mean_absolute_error, KMeans

    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    from sklearn.compose import ColumnTransformer  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.linear_model import ElasticNetCV  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import r2_score, mean_absolute_error  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore

    # 將導入的模組/類別掛到全域，讓原本的 helper functions 不用大改
    globals()["np"] = np
    globals()["pd"] = pd
    globals()["ColumnTransformer"] = ColumnTransformer
    globals()["Pipeline"] = Pipeline
    globals()["StandardScaler"] = StandardScaler
    globals()["OneHotEncoder"] = OneHotEncoder
    globals()["SimpleImputer"] = SimpleImputer
    globals()["RandomForestRegressor"] = RandomForestRegressor
    globals()["ElasticNetCV"] = ElasticNetCV
    globals()["train_test_split"] = train_test_split
    globals()["r2_score"] = r2_score
    globals()["mean_absolute_error"] = mean_absolute_error
    globals()["KMeans"] = KMeans
# ==== 固定版四群「人才羅盤」模型卡（先給預設文字，你之後可以改） ====

CLUSTER_CARDS: Dict[int, Dict[str, Any]] = {
    0: {
        "key": "growth_high_engagement",
        "name": "高投入成長型",
        "short_title": "主動學習，還有很大成長空間",
        "summary": "通常學習動機高、訓練參與度佳，績效已不錯但仍在上升軌道。",
        "dev_focus": [
            "提供具挑戰性的專案或輪調機會",
            "設計明確的晉升／職涯晉級路徑",
            "定期回饋，協助調整學習重點"
        ],
        "risk_alert": [
            "避免長期高投入卻升遷停滯，導致動機下滑",
            "注意工作負荷是否過高，影響工作生活平衡"
        ],
        "suggestions": [
            "安排一位資深導師，協助規劃未來 1–2 年的成長目標",
            "讓他參與跨部門專案，擴大影響範圍",
            "搭配明確的技能認證或徽章制度，強化成就感"
        ]
    },
    1: {
        "key": "steady_veteran",
        "name": "穩健資深型",
        "short_title": "年資深、表現穩定的關鍵支柱",
        "summary": "在組織年資較長、績效穩定，是團隊中的穩定力量與知識來源。",
        "dev_focus": [
            "強化知識傳承與教練角色",
            "協助更新技能，避免與新技術脫節",
            "鼓勵參與制度／流程優化專案"
        ],
        "risk_alert": [
            "留意是否出現動能下滑或對變革抗拒",
            "避免只被視為『穩定人力』而缺乏成長機會"
        ],
        "suggestions": [
            "設計『資深員工 mentor 計畫』，由他帶新同仁",
            "邀請參與內部訓練課程授課或共備",
            "在績效對話中加入『傳承與影響力』的指標"
        ]
    },
    2: {
        "key": "high_pressure_risky",
        "name": "高壓高風險型",
        "short_title": "負荷高、績效可能兩極，需要風險管理",
        "summary": "常出現高工時、高壓力或頻繁加班，績效有時亮眼、有時不穩定。",
        "dev_focus": [
            "調整工作負荷與角色定位，避免長期過勞",
            "引入壓力管理與心理資源",
            "明確設定優先順序與可被拒絕的界線"
        ],
        "risk_alert": [
            "倦怠風險高，可能突然離職或表現驟降",
            "容易影響團隊氛圍，讓壓力文化擴散"
        ],
        "suggestions": [
            "檢視手上的任務與 KPI，協助刪減非關鍵工作",
            "提供彈性工時或休假安排，讓他有恢復空間",
            "HR 定期 1:1 check-in，追蹤壓力與健康狀況"
        ]
    },
    3: {
        "key": "emerging_talent",
        "name": "新秀潛力型",
        "short_title": "年資較短、潛力明顯，需要被好好栽培",
        "summary": "剛加入或年資尚短，表現已展現潛力，但仍在摸索階段。",
        "dev_focus": [
            "快速補齊核心技能與制度知識",
            "建立清楚的期望與回饋頻率",
            "讓他有小規模試錯與嘗試的空間"
        ],
        "risk_alert": [
            "若缺乏指導與回饋，容易迷惘或喪失信心",
            "太快壓上關鍵任務，可能壓力過大"
        ],
        "suggestions": [
            "安排入職後 3–6 個月的結構化培訓與 check-point",
            "在績效對話中多關注『學習曲線』而非單次結果",
            "搭配師徒制或同儕 buddy，提供日常支持"
        ]
    },
}



try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False


app = FastAPI(title="AI Talent Predictor API", version="0.2")

# 讓 Chrome DevTools 的探測請求安靜一點（不影響功能）
@app.get("/.well-known/appspecific/{path:path}")
def _chrome_devtools_noise(path: str):
    return Response(status_code=204)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"ok": True}

# ===== 全域狀態：存模型、前處理器、欄位資訊 =====
MODEL_STATE: Dict[str, Any] = {
    "pipe_rf": None,
    "preprocessor": None,
    "numeric_cols": None,
    "categorical_cols": None,
    "feature_cols": None,
    "defaults_all": None,
    "shap_explainer": None,
    "shap_feature_names": None,
    "top_features": None,   # 已存在：給 Demo 頁用
    "metrics": None,        # 已存在
    "kmeans": None,         # 👈 新增：k=4 分群模型
    "perf_quantiles": None, # 👈 新增：用來判斷高/中/低績效 band
    # 🔹 新增：保留完整特徵資料和 y，給自由分群用
    "X_all": None,
    "y_all": None,
    "df_for_cluster": None,   # 👈 新增：給自由分群沙盒用
}




class PredictRequest(BaseModel):
    features: Dict[str, Any]

class ClusterSandboxRequest(BaseModel):
    k: int


class TalentCompassRequest(BaseModel):
    features: Dict[str, Any]

class ClusterPlayRequest(BaseModel):
    k: int = 4   # 預設 4 群



# ===== 共用小工具 =====
def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取 CSV：{e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV 內容為空")
    return df


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def build_preprocessor(df_features: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in df_features.columns if is_numeric_series(df_features[c])]
    cat_cols = [c for c in df_features.columns if c not in num_cols]

    num_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        [
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ]
    )
    return pre, num_cols, cat_cols


# ===== 模型訓練主流程（照你 Step2 改成函式版） =====
def train_model(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    # Render 免費/小機器很容易因為資料太大或模型太重被打爆（timeout / OOM → 502）。
    # 這裡先做「保守限速」：資料抽樣 + 輕量模型參數，讓訓練能在短時間內完成。
    import time
    t0 = time.time()

    MAX_ROWS = 6000
    n_rows_raw = int(len(df))
    if n_rows_raw > MAX_ROWS:
        # 抽樣後把 df 覆蓋掉，避免整份大表一直留在記憶體裡
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    n_rows_used = int(len(df))

    target_col = "Performance_Score"
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail="資料中找不到 Performance_Score 欄位")

    # 1) 避免洩漏：丟掉 ID / 日期 / 目標 / 離職、晉升、cluster 等欄位
    drop_cols: List[str] = []
    for col in df.columns:
        lower = col.lower()
        if "id" in lower or "date" in lower:
            drop_cols.append(col)

    drop_cols.extend(
        [
            target_col,
            "Resigned",
            "Promotions",
            "Promotion_Last_3_Years",
            "cluster_kmeans_v2",
            "cluster_kmeans",
            "y_true",
            "y_pred",
        ]
    )
    drop_cols = sorted(set([c for c in drop_cols if c in df.columns]))

    y = df[target_col].astype(float)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="移除 ID / 目標欄位後沒有剩下任何特徵可用")

    X = df[feature_cols].copy()

    # 給自由分群沙盒用：保留特徵 + 目標
    MODEL_STATE["df_for_cluster"] = df[feature_cols + [target_col]].copy()

    # 🔹 新增：把完整 X / y 存起來，讓後面自由分群可以用同一份資料
    MODEL_STATE["X_all"] = X.copy()
    MODEL_STATE["y_all"] = y.copy()

    print(
        f"[train] start rows={n_rows_raw} used={n_rows_used} features={len(feature_cols)} test_size={test_size}",
        flush=True,
    )


    # 2) 前處理 + 切訓練 / 測試
    pre, num_cols, cat_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 3) 模型：Render 輕量版只留 RandomForest（避免 CV 類模型在 Render 上跑太久/吃太多記憶體）
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )

    pipe_rf = Pipeline([("pre", pre), ("rf", rf)])
    pipe_rf.fit(X_train, y_train)

    # 4) 評估指標
    pred_rf_train = pipe_rf.predict(X_train)
    pred_rf_test = pipe_rf.predict(X_test)
    metrics = {
        "rf": {
            "R2_train": float(r2_score(y_train, pred_rf_train)),
            "R2_test": float(r2_score(y_test, pred_rf_test)),
            "MAE_test": float(mean_absolute_error(y_test, pred_rf_test)),
        }
    }


    # === 新增 A：用整體資料 + 前處理器 訓練 k=4 分群 ===
    fitted_pre = pipe_rf.named_steps["pre"]
    X_all_trans = fitted_pre.transform(X)   # X 是全資料，不只訓練集
    kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_4.fit(X_all_trans)

    # === 新增 B：計算績效分層用的分位數（低/中/高） ===
    q_low = float(np.quantile(y_train, 0.33))
    q_high = float(np.quantile(y_train, 0.66))
    perf_quantiles = {"low": q_low, "high": q_high}


    # 5) 為每個欄位算「預設值」（中位數 / 眾數），給預測頁用
    defaults_all: Dict[str, Any] = {}
    for col in feature_cols:
        s = X[col]
        if is_numeric_series(s):
            defaults_all[col] = float(s.median())
        else:
            mode = s.mode()
            defaults_all[col] = mode.iloc[0] if not mode.empty else ""

    # 6) SHAP 全域重要度（聚合到「原始欄位」層級）
    shap_explainer = None
    shap_feature_names: List[str] = []
    shap_global_agg: List[Dict[str, Any]] = []

    # 不管有沒有 SHAP，都先把「轉換後特徵」名稱取出，方便做聚合

    fitted_pre = pipe_rf.named_steps["pre"]

    fitted_rf = pipe_rf.named_steps["rf"]


    try:

        feature_names = list(fitted_pre.get_feature_names_out())

    except Exception:

        # 退一步：若 sklearn 版本不支援，就只能以索引代稱

        fi_len = getattr(fitted_rf, "feature_importances_", np.array([])).shape[0]

        feature_names = [f"f{i}" for i in range(fi_len)]


    def _base_feature_name(trans_name: str) -> str:

        # ColumnTransformer 的輸出常見格式：num__Age / cat__Department_Sales

        if trans_name.startswith("num__"):

            return trans_name.split("__", 1)[1]

        if trans_name.startswith("cat__"):

            sub = trans_name.split("__", 1)[1]

            # OneHotEncoder：<col>_<category>

            # 用「最長前綴匹配」避免欄位名本身有底線被切壞

            best = None

            for c in cat_cols:

                prefix = f"{c}_"

                if sub == c or sub.startswith(prefix):

                    if best is None or len(c) > len(best):

                        best = c

            return best or sub

        return trans_name


    if HAS_SHAP:

        # ✅ 完整版：用 SHAP 算全域重要度（較準，但套件很大）

        sample_size = min(800, len(X_train))

        rng = np.random.RandomState(42)

        sample_idx = rng.choice(X_train.index, size=sample_size, replace=False)

        X_train_sample = X_train.loc[sample_idx]

        X_train_trans = fitted_pre.transform(X_train_sample)


        shap_explainer = shap.TreeExplainer(fitted_rf)

        shap_values = shap_explainer.shap_values(X_train_trans)


        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        agg: Dict[str, float] = {}

        for fname, val in zip(feature_names, mean_abs_shap):

            base = _base_feature_name(str(fname))

            agg[base] = agg.get(base, 0.0) + float(val)


        shap_global_agg = [

            {"feature": base, "mean_abs_shap": float(val)}

            for base, val in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)

        ][:15]


        shap_feature_names = feature_names


    else:

        # ✅ 輕量版：不用 SHAP，改用 RandomForest 的 feature_importances_

        importances = getattr(fitted_rf, "feature_importances_", None)

        if importances is None:

            importances = np.zeros(len(feature_names), dtype=float)


        agg: Dict[str, float] = {}

        for fname, val in zip(feature_names, importances):

            base = _base_feature_name(str(fname))

            agg[base] = agg.get(base, 0.0) + float(val)


        shap_global_agg = [

            {"feature": base, "mean_abs_shap": float(val)}

            for base, val in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)

        ][:15]

    # 7) 組合前端會用到的欄位資訊（含提示）
    top_features_with_defaults: List[Dict[str, Any]] = []

    for row in shap_global_agg:
        fname = row["feature"]
        row_out: Dict[str, Any] = dict(row)
        default_val = defaults_all.get(fname)
        row_out["default"] = default_val

        if fname in num_cols:
            row_out["value_type"] = "number"
            # 給一個大概範圍的提示
            hint = None
            try:
                s = df[fname]
                p5 = float(np.nanpercentile(s, 5))
                p95 = float(np.nanpercentile(s, 95))
                med = float(np.nanmedian(s))
                hint = f"建議輸入數字，常見範圍約 {p5:.1f}–{p95:.1f}（中位數 {med:.1f}）"
            except Exception:
                hint = "請輸入數字（單位同原始資料）。"
            row_out["hint"] = hint
        else:
            row_out["value_type"] = "text"
            hint = None
            try:
                if fname in df.columns:
                    s = df[fname].astype(str)
                    top_vals = s.value_counts().head(4).index.tolist()
                    if top_vals:
                        joined = " / ".join(map(str, top_vals))
                        hint = f"建議輸入資料中出現過的文字，例如：{joined}"
            except Exception:
                pass
            if not hint:
                hint = "請輸入文字（盡量使用資料集中出現過的類別）。"
            row_out["hint"] = hint

        top_features_with_defaults.append(row_out)

    # 8) 把模型相關物件存到全域狀態，方便 /api/predict 與 /api/model_summary 使用
    MODEL_STATE["pipe_rf"] = pipe_rf
    MODEL_STATE["preprocessor"] = pipe_rf.named_steps["pre"]
    MODEL_STATE["numeric_cols"] = num_cols
    MODEL_STATE["categorical_cols"] = cat_cols
    MODEL_STATE["feature_cols"] = feature_cols
    MODEL_STATE["defaults_all"] = defaults_all
    MODEL_STATE["shap_explainer"] = shap_explainer
    MODEL_STATE["shap_feature_names"] = shap_feature_names
    MODEL_STATE["top_features"] = top_features_with_defaults  # ✅ 現在變數已經先算好
    MODEL_STATE["metrics"] = metrics

    # 👇 新增：存 k=4 與分位數
    MODEL_STATE["kmeans"] = kmeans_4
    MODEL_STATE["perf_quantiles"] = perf_quantiles

    secs = time.time() - t0
    print(f"[train] done used={n_rows_used} in {secs:.1f}s", flush=True)

    return {
        "target_col": target_col,
        "n_rows_raw": n_rows_raw,
        "n_rows_used": n_rows_used,
        "n_features": int(len(feature_cols)),
        "metrics": metrics,
        "top_features": top_features_with_defaults,
        "has_shap": HAS_SHAP,
    }



# ===== API：上傳 CSV → 訓練模型 =====
@app.post("/api/train_model")
async def api_train_model(
    file: UploadFile = File(...),
    test_size: float = Form(0.2),
):
    _lazy_import_ml()  # lazy load numpy/pandas/sklearn
    try:
        df = read_csv_upload(file)
        result = train_model(df, test_size=test_size)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        print("[train] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"訓練失敗：{e}")


# ===== API：單一樣本預測 =====
@app.post("/api/predict")
async def api_predict(req: PredictRequest = Body(...)):
    _lazy_import_ml()  # lazy load numpy/pandas/sklearn
    if MODEL_STATE["pipe_rf"] is None:
        raise HTTPException(
            status_code=400,
            detail="尚未訓練模型，請先在預測頁面上傳資料並執行「建立模型」",
        )

    pipe_rf: Pipeline = MODEL_STATE["pipe_rf"]
    pre = MODEL_STATE["preprocessor"]
    feature_cols: List[str] = MODEL_STATE["feature_cols"]
    defaults_all: Dict[str, Any] = MODEL_STATE["defaults_all"] or {}
    shap_explainer = MODEL_STATE["shap_explainer"]
    shap_feature_names: List[str] = MODEL_STATE["shap_feature_names"] or []

    # 以訓練資料的「預設值」當成 base 個案，再用使用者輸入覆蓋
    row_data: Dict[str, Any] = {}
    for col in feature_cols:
        row_data[col] = defaults_all.get(col)

    for k, v in req.features.items():
        if k in row_data:
            row_data[k] = v

    X_new = pd.DataFrame([row_data], columns=feature_cols)

    y_pred = float(pipe_rf.predict(X_new)[0])

    # 影響因子（Top10）
    # - 如果有 SHAP：回傳「個案」的 SHAP Top 10（較精準，但較吃資源）
    # - 如果沒有 SHAP：回傳「全域」的重要度 Top 10（輕量版 fallback）
    explain_details: List[Dict[str, Any]] = []

    if HAS_SHAP and shap_explainer is not None and shap_feature_names:
        X_new_trans = pre.transform(X_new)
        shap_values = shap_explainer.shap_values(X_new_trans)[0]
        abs_vals = np.abs(shap_values)
        order = np.argsort(-abs_vals)
        top_k = min(10, len(order))
        for idx in order[:top_k]:
            explain_details.append(
                {
                    "feature": shap_feature_names[idx],
                    "shap_value": float(shap_values[idx]),
                    "abs_shap": float(abs_vals[idx]),
                }
            )
    else:
        # fallback：用訓練階段算好的「全域重要度」
        top_features = MODEL_STATE.get("top_features") or []
        for rank, row in enumerate(top_features[:10], start=1):
            imp = float(row.get("mean_abs_shap") or 0.0)
            explain_details.append(
                {
                    "feature": row.get("feature", f"feature_{rank}"),
                    "shap_value": imp,   # 前端沿用同一欄位顯示
                    "abs_shap": imp,
                }
            )

    return {
        "prediction": y_pred,
        "shap_top_contrib": explain_details,
    }


from fastapi import Body

# （前面 CLUSTER_CARDS、BAND_THRESHOLDS 等保持不動）

@app.post("/api/talent_compass_predict")
async def api_talent_compass_predict(req: PredictRequest = Body(...)):
    _lazy_import_ml()  # lazy load numpy/pandas/sklearn
    if MODEL_STATE["pipe_rf"] is None:
        raise HTTPException(
            status_code=400,
            detail="尚未訓練模型，請先在預測頁面上傳資料並執行「建立模型」",
        )

    pipe_rf: Pipeline = MODEL_STATE["pipe_rf"]
    pre = MODEL_STATE["preprocessor"]
    feature_cols: List[str] = MODEL_STATE["feature_cols"] or []
    defaults_all: Dict[str, Any] = MODEL_STATE["defaults_all"] or {}

    # 1) 組一列資料（用 defaults 當底，再用使用者輸入覆蓋）
    row_data = {col: defaults_all.get(col) for col in feature_cols}
    for k, v in req.features.items():
        if k in row_data:
            row_data[k] = v

    X_new = pd.DataFrame([row_data], columns=feature_cols)

    # 2) 先用 RF 預測績效分數
    score = float(pipe_rf.predict(X_new)[0])

    # 3) 用訓練時算好的分位數，決定 high / medium / low
    perf_q = MODEL_STATE.get("perf_quantiles") or {}
    q_low = perf_q.get("low")
    q_high = perf_q.get("high")

    if q_low is not None and q_high is not None:
        if score >= q_high:
            band_key = "high"
        elif score >= q_low:
            band_key = "medium"
        else:
            band_key = "low"
    else:
        # 如果沒算到分位數，就用固定門檻當備援
        if score >= 4.2:
            band_key = "high"
        elif score >= 3.4:
            band_key = "medium"
        else:
            band_key = "low"

    BAND_LABEL_ZH = {
        "high": "高績效帶",
        "medium": "中等績效帶",
        "low": "需關注績效帶",
    }

    # 4) 用 k-means 分群（真正對應到你研究的 4 群）
    kmeans: KMeans | None = MODEL_STATE.get("kmeans")
    if kmeans is not None:
        X_new_trans = pre.transform(X_new)
        cluster_id = int(kmeans.predict(X_new_trans)[0])  # 0~3
    else:
        cluster_id = 0  # 備援

    # 5) 用整數 cluster_id 去拿對應的人才卡
    card = CLUSTER_CARDS.get(cluster_id, {})

    dev_focus = card.get("dev_focus", [])
    # 注意：原本 key 叫 risk_alert / suggestions，這邊幫你轉一下
    risk_alerts = card.get("risk_alert", [])
    hr_tips = card.get("suggestions", [])

    payload = {
        "performance_score": score,
        "performance_band": band_key,
        "performance_level": BAND_LABEL_ZH.get(band_key, "尚未標定等級"),

        "cluster_id": cluster_id,
        "cluster_name": card.get("name", f"第 {cluster_id + 1} 群"),
        "cluster_short_title": card.get("short_title", ""),
        "cluster_summary": card.get("summary", ""),

        "dev_focus": dev_focus,
        "risk_alerts": risk_alerts,
        "hr_tips": hr_tips,

        "cluster_card": card,
    }
    return payload



@app.post("/api/cluster_sandbox")
async def api_cluster_sandbox(req: ClusterSandboxRequest = Body(...)):
    _lazy_import_ml()  # lazy load numpy/pandas/sklearn
    """
    自由分群遊樂場用：
    - 使用 Demo 已訓練好的資料與前處理
    - 只調整 k（群數），回傳各群人數、平均績效與「相對整體的特徵輪廓」
    """
    if MODEL_STATE["df_for_cluster"] is None or MODEL_STATE["preprocessor"] is None:
        raise HTTPException(
            status_code=400,
            detail="尚未訓練模型或尚未載入原始資料，請先在 Demo 頁上傳資料並建立模型。",
        )

    k = max(2, int(req.k))
    df_cluster: pd.DataFrame = MODEL_STATE["df_for_cluster"].copy()
    feature_cols: List[str] = MODEL_STATE["feature_cols"] or []
    pre = MODEL_STATE["preprocessor"]

    if not feature_cols:
        raise HTTPException(status_code=400, detail="找不到可用特徵欄位。")

    X_all = df_cluster[feature_cols]
    y_all = df_cluster["Performance_Score"].astype(float)
    n_samples = len(df_cluster)

    # 先用既有前處理轉換，再做 k-means 分群
    X_all_trans = pre.transform(X_all)

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto",
    )
    labels = km.fit_predict(X_all_trans)
    df_cluster["cluster"] = labels

    overall_mean = float(y_all.mean())

    # 🔹 只針對「數值特徵」做輪廓比較
    numeric_cols: List[str] = MODEL_STATE["numeric_cols"] or []
    numeric_cols = [c for c in numeric_cols if c in df_cluster.columns]

    if numeric_cols:
        global_feature_means = (
            df_cluster[numeric_cols].mean(numeric_only=True).to_dict()
        )
    else:
        global_feature_means = {}

    clusters_out: List[Dict[str, Any]] = []

    # 依各群平均績效排序後再產出說明
    cluster_stats: List[Tuple[int, float]] = []
    for cid in range(k):
        mask = df_cluster["cluster"] == cid
        if mask.sum() == 0:
            continue
        mean_perf = float(df_cluster.loc[mask, "Performance_Score"].mean())
        cluster_stats.append((cid, mean_perf))

    # 依平均績效由高到低排序
    cluster_stats.sort(key=lambda x: x[1], reverse=True)

    for rank_idx, (cid, mean_perf) in enumerate(cluster_stats):
        mask = df_cluster["cluster"] == cid
        sub = df_cluster.loc[mask]
        n_c = int(mask.sum())
        prop = n_c / n_samples if n_samples > 0 else 0.0
        std_perf = float(sub["Performance_Score"].std(ddof=0) or 0.0)

        # 平均分數在整體的粗略百分位
        mean_percentile = float((y_all < mean_perf).mean()) if n_samples > 0 else 0.0

        if rank_idx == 0:
            rank_label = "整體高績效輪廓"
            comment = "這一群的平均績效最高，適合深入觀察其共同特徵，作為關鍵人才輪廓的參考。"
        elif rank_idx == len(cluster_stats) - 1:
            rank_label = "相對低績效輪廓"
            comment = "平均績效明顯低於其他群，適合搭配訓練與工作設計，思考如何拉抬表現。"
        else:
            rank_label = "中間績效輪廓"
            comment = "平均績效介於兩端之間，可再依工作內容、年資等變項拆解子輪廓。"

        # 🔹 本群各數值特徵平均，與整體比較
        if numeric_cols:
            feature_means = (
                sub[numeric_cols].mean(numeric_only=True).to_dict()
            )
        else:
            feature_means = {}

        diff_list: List[Dict[str, Any]] = []
        for col in numeric_cols:
            g = global_feature_means.get(col)
            c_mean = feature_means.get(col)
            if g is None or c_mean is None or pd.isna(g) or pd.isna(c_mean):
                continue
            diff_val = float(c_mean - g)
            diff_list.append(
                {
                    "feature": col,
                    "cluster_mean": float(c_mean),
                    "global_mean": float(g),
                    "diff": diff_val,
                }
            )

        diff_list.sort(key=lambda d: abs(d["diff"]), reverse=True)
        top_feature_diff = diff_list[:5]

        clusters_out.append(
            {
                "cluster_id": cid,
                "display_name": f"第 {cid + 1} 群",
                "n_samples": n_c,
                "proportion": float(prop),
                "mean_performance": mean_perf,
                "std_performance": std_perf,
                "mean_percentile": mean_percentile,
                "rank_label": rank_label,
                "comment": comment,
                "feature_means": feature_means,      # ⭐ 新增：每群全部數值特徵平均
                "top_feature_diff": top_feature_diff,  # ⭐ 給前端用
            }
        )

    return {
        "k": k,
        "n_samples": n_samples,
        "overall_mean_performance": overall_mean,
        "feature_means_global": global_feature_means,  # ⭐ 全體 baseline
        "clusters": clusters_out,
    }



    
@app.post("/api/cluster_playground")
async def api_cluster_playground(req: ClusterPlayRequest = Body(...)):
    _lazy_import_ml()  # lazy load numpy/pandas/sklearn
    """
    自由版分群遊樂場：
    - 使用目前訓練好的資料（X_all / y_all）
    - 只用數值欄位做 KMeans 分群
    - 回傳每一群的人數、比例、平均 Performance_Score、以及差異最大的一些數值特徵
    """
    if MODEL_STATE["X_all"] is None or MODEL_STATE["y_all"] is None:
        raise HTTPException(
            status_code=400,
            detail="尚未訓練模型，請先在研究版頁面上傳資料並建立模型。",
        )

    X_all: pd.DataFrame = MODEL_STATE["X_all"]
    y_all: pd.Series = MODEL_STATE["y_all"]

    k = int(req.k)
    if k < 2 or k > 8:
        raise HTTPException(
            status_code=400,
            detail="群數 k 建議介於 2～8（太多群可讀性會變差）。",
        )

    num_cols: List[str] = MODEL_STATE["numeric_cols"] or []
    if not num_cols:
        raise HTTPException(status_code=400, detail="找不到可用的數值欄位，無法分群。")

    # 只用數值欄位做分群，避免文字欄位的處理問題
    X_num = X_all[num_cols].copy()

    # 簡單版前處理：缺值補中位數＋標準化 → KMeans
    cluster_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            # 用整數 n_init，避免某些 sklearn 版本不支援 "auto"
            ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
        ]
    )
    cluster_pipe.fit(X_num)

    labels = cluster_pipe.named_steps["kmeans"].labels_
    n_total = len(X_num)
    overall_means = X_num.mean(axis=0)
    perf_overall = float(y_all.mean())

    clusters_out: List[Dict[str, Any]] = []
    for cid in range(k):
        mask = labels == cid
        n_cluster = int(mask.sum())
        if n_cluster == 0:
            continue

        frac = float(n_cluster / n_total)
        cluster_means = X_num[mask].mean(axis=0)
        perf_mean = float(y_all[mask].mean())

        # 找出「此群 vs 全體」差異最大的前 5 個特徵
        diff = (cluster_means - overall_means).abs().sort_values(ascending=False)
        top_features: List[Dict[str, Any]] = []
        for fname in diff.index[:5]:
            top_features.append(
                {
                    "feature": fname,
                    "cluster_mean": float(cluster_means[fname]),
                    "overall_mean": float(overall_means[fname]),
                }
            )

        clusters_out.append(
            {
                "cluster_id": cid,
                "size": n_cluster,
                "ratio": frac,
                "performance_mean": perf_mean,
                "performance_overall": perf_overall,
                "top_diff_features": top_features,
            }
        )

    return {
        "k": k,
        "n_rows": n_total,
        "numeric_features": num_cols,
        "clusters": clusters_out,
    }




@app.get("/api/model_summary")
async def api_model_summary():
    """
    回傳目前已訓練模型的摘要資訊：
    - metrics：RF / ElasticNet 的 R2 / MAE
    - top_features：前幾個 SHAP 重要特徵（含預設值與提示）
    """
    if MODEL_STATE["pipe_rf"] is None:
        raise HTTPException(
            status_code=400,
            detail="尚未訓練模型，請先到「預測 Demo 頁」上傳資料並建立模型。",
        )

    return {
        "has_model": True,
        "has_shap": HAS_SHAP,
        "metrics": MODEL_STATE.get("metrics") or {},
        "top_features": MODEL_STATE.get("top_features") or [],
        "feature_cols": MODEL_STATE.get("feature_cols") or [],
    }

# ===== 靜態頁面（Render 部署友善：同一個服務同時提供 API + HTML）=====
_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"

if _STATIC_DIR.exists():
    # html=True 會讓 / 自動回傳 index.html
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

