# AI Talent Predictor (Render 輕量版)

這版主打：
- 同一個 Render Web Service 同時提供 **API** + **純 HTML 前端**
- 不安裝 SHAP（省記憶體/安裝時間），改用 RandomForest 的 `feature_importances_` 做全域重要度（前端仍可顯示 Top Features）
- 保留你現有的三頁：`/`(index)、`/predict.html`、`/sandbox.html`

## 專案結構
```
render_lite/
  main.py
  requirements.txt
  render.yaml
  static/
    index.html
    predict.html
    sandbox.html
```

## Render 部署（最簡單做法）
1. 把這個資料夾推到 GitHub
2. Render → New → Web Service → 選 repo
3. Build Command：
   - `pip install -r requirements.txt`
4. Start Command：
   - `uvicorn main:app --host 0.0.0.0 --port $PORT`

部署成功後，直接打開：
- 首頁：`https://<你的 Render 網址>/`
- Demo：`https://<你的 Render 網址>/predict.html`
- 分群：`https://<你的 Render 網址>/sandbox.html`

## 本機測試
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## 注意
- Render 的免費方案會睡眠，第一次打開可能會慢一點（正常）。
- 若你「一定要」個案 SHAP 貢獻表（/api/predict 的 shap_top_contrib），就需要把 `shap` 加回 requirements，但會變重。
