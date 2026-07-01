"""Model Registry page — queries MLflow directly + shows calibration artifact."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv

load_dotenv()
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

st.set_page_config(page_title="Model Registry", page_icon="📊", layout="wide")
sidebar_lang_toggle()

st.title(L("📊 Model Kayıt Deposu (MLflow)", "📊 Model Registry (MLflow)"))
st.caption(L(
    "Her train çalışması MLflow'a params + metrics + model artifact kayıtlar. "
    "Serving `saved_models/` içindeki en yeni joblib'i yükler ve MLflow'daki registered "
    "version'a eşler.",
    "Every training run logs params, metrics, and a model artifact to MLflow. "
    "Serving loads the newest joblib from `saved_models/` and matches it to a registered version.",
))

with st.expander(L("ℹ️ Bu sayfa ne yapıyor?", "ℹ️ What does this page do?")):
    st.markdown(L(
        """
        MLflow, ML dünyasında "her deneyimin muhasebe defteri":

        - **Registered Models**: Adı olan, versiyonlu model kayıtları. `propensity_model v3`,
          `ltv_model v12` gibi. Her train'de yeni version.
        - **Experiments & Runs**: Her train bir "run". İçerikleri: hyperparameters, metrikler,
          model artifact, ne zaman + hangi kod dosyasından geldiği.
        - **Metric Trend**: Aynı metriği farklı run'larda karşılaştırmak.
        - **Calibration Curve**: v1 (bozuk) vs v2 (kalibre edilmiş) reliability diagram.

        **Neden önemli**: 3 hafta sonra "bu tahmin hangi modeli kullandı?" sorusunun cevabı
        buradadır. Reproducibility ve trust MLOps'un temeli.
        """,
        """
        MLflow is the ledger of your ML experiments:

        - **Registered Models**: Named, versioned model records. `propensity_model v3`,
          `ltv_model v12`, etc. New version every training run.
        - **Experiments & Runs**: Each training call is one "run" containing params, metrics,
          the model artifact, and provenance info.
        - **Metric Trend**: Compare the same metric across runs.
        - **Calibration Curve**: v1 (broken) vs v2 (calibrated) reliability diagram.

        **Why it matters**: In three weeks, "which model produced this prediction?" is
        answered here. Reproducibility and trust are the core of MLOps.
        """,
    ))


# ── Fetch ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_registered_models() -> pd.DataFrame:
    r = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search", timeout=5)
    r.raise_for_status()
    rows = []
    for m in r.json().get("registered_models", []):
        for v in m.get("latest_versions", []):
            rows.append({
                "model":   m["name"],
                "version": int(v["version"]),
                "status":  v.get("status", ""),
                "stage":   v.get("current_stage", ""),
                "run_id":  v.get("run_id", ""),
                "source":  v.get("source", "")[:70],
            })
    return pd.DataFrame(rows).sort_values(["model", "version"], ascending=[True, False])


@st.cache_data(ttl=30)
def fetch_experiments() -> list[dict]:
    r = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
        json={"max_results": 20}, timeout=5,
    )
    r.raise_for_status()
    return r.json().get("experiments", [])


@st.cache_data(ttl=30)
def fetch_runs_for_experiment(exp_id: str, max_results: int = 20) -> pd.DataFrame:
    r = requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
        json={"experiment_ids": [exp_id], "max_results": max_results,
              "order_by": ["start_time DESC"]},
        timeout=5,
    )
    r.raise_for_status()
    rows = []
    for run in r.json().get("runs", []):
        info = run["info"]
        data = run.get("data", {})
        # MLflow REST 3.x returns metrics/tags as arrays of {key, value} dicts,
        # not native dicts. Normalise both here so downstream code stays clean.
        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        tags    = {t["key"]: t["value"] for t in data.get("tags",    [])}
        rows.append({
            "run_id":   info["run_id"][:12],
            "status":   info["status"],
            "run_name": tags.get("mlflow.runName", ""),
            **{k: metrics.get(k, None) for k in
               ["test_auc", "test_brier", "test_calibration_max_delta",
                "f1_optimal_threshold", "test_r2", "test_mae"]},
            "start_time": pd.to_datetime(info["start_time"], unit="ms"),
        })
    return pd.DataFrame(rows)


try:
    models_df = fetch_registered_models()
except Exception as ex:
    st.error(f"MLflow unreachable at {MLFLOW_URL}: {ex}")
    st.stop()


st.subheader(L("Kayıtlı Modeller", "Registered Models"))
st.caption(L(
    "Her train'de yeni bir version satırı oluşur. Serving en yeni versiyonu kullanır.",
    "A new version row appears on each train. Serving uses the newest.",
))
st.dataframe(models_df, use_container_width=True, hide_index=True)


st.divider()
st.subheader(L("Deneyler ve Son Çalışmalar", "Experiments & Recent Runs"))
st.caption(L(
    "İki experiment var: propensity + LTV. Her train bir 'run'.",
    "Two experiments: propensity + LTV. Each training call is one run.",
))

exps = fetch_experiments()
if not exps:
    st.info(L("Experiment bulunamadı.", "No experiments found."))
else:
    exp_names = [e["name"] for e in exps if e["name"] != "Default"]
    picked_exp = st.selectbox("Experiment", exp_names)
    exp_id = next(e["experiment_id"] for e in exps if e["name"] == picked_exp)

    runs = fetch_runs_for_experiment(exp_id)
    if runs.empty:
        st.info(f"No runs in `{picked_exp}`.")
    else:
        st.dataframe(
            runs.assign(
                test_auc=lambda d: d["test_auc"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
                test_brier=lambda d: d["test_brier"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
                test_calibration_max_delta=lambda d: d["test_calibration_max_delta"].apply(
                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
                f1_optimal_threshold=lambda d: d["f1_optimal_threshold"].apply(
                    lambda v: f"{v:.3f}" if pd.notna(v) else "—"),
                test_r2=lambda d: d["test_r2"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—"),
                test_mae=lambda d: d["test_mae"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—"),
            ),
            use_container_width=True, hide_index=True,
        )

        st.subheader(L("Metrik Trend", "Metric Trend Across Runs"))
        st.caption(L(
            "Aynı metriğin farklı run'larda nasıl değiştiğini gör.",
            "See how a metric evolves across training runs.",
        ))
        numeric_cols = ["test_auc", "test_brier", "test_calibration_max_delta",
                        "test_r2", "test_mae"]
        available = [c for c in numeric_cols if runs[c].notna().any()]
        if available:
            metric_pick = st.selectbox(L("Metrik", "Metric"), available)
            df_trend = runs.dropna(subset=[metric_pick]).sort_values("start_time")
            fig = px.line(
                df_trend, x="start_time", y=metric_pick, markers=True,
                title=L(f"{metric_pick} zaman içinde", f"{metric_pick} over time"),
                labels={"start_time": L("Run zamanı", "Run time")},
            )
            fig.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)


st.divider()
st.subheader(L("Kalibrasyon Eğrisi (v1 vs v2)", "Calibration Curve (v1 vs v2)"))
st.caption(L(
    "v1 → sızıntılı + scale_pos_weight ile eğitilmiş (olasılıklar bozuk). "
    "v2 → isotonic calibration ile düzeltilmiş (olasılıklar güvenilir). "
    "Bu grafik dürüstlük belgemiz.",
    "v1 → leaky + scale_pos_weight (probabilities broken). "
    "v2 → isotonic-calibrated (probabilities trustworthy). "
    "This chart is our honesty artifact.",
))

calib_path = Path("saved_models/calibration_curve.png")
if calib_path.exists():
    st.image(
        str(calib_path),
        caption=L(
            "Diyagonal = mükemmel kalibrasyon.",
            "Diagonal = perfect calibration.",
        ),
        use_column_width=True,
    )
else:
    st.warning(L(
        "`saved_models/calibration_curve.png` bulunamadı. "
        "Üretmek için: `uv run python -m scripts.audit_models`",
        "`saved_models/calibration_curve.png` not found. "
        "Run `uv run python -m scripts.audit_models` to generate it.",
    ))


st.divider()
st.markdown(L(
    """
    ### Serving hangi model versiyonunu seçiyor?
    ```python
    # src/ml/inference.py
    prop_path = max(glob.glob("saved_models/propensity_v*.joblib"), key=os.path.getmtime)
    ```
    Diskteki en yeni joblib kazanır ve dosya adı MLflow run_id prefix'ini içerir. Bu bilinçli:
    MLflow 3.x'in `skops` güvenlik katmanı XGBoost+CalibratedClassifierCV için flaky, o yüzden
    joblib source of truth, MLflow metadata mirror.
    """,
    """
    ### How serving picks a model version
    ```python
    # src/ml/inference.py
    prop_path = max(glob.glob("saved_models/propensity_v*.joblib"), key=os.path.getmtime)
    ```
    The newest joblib on disk wins, and its filename embeds the MLflow run_id prefix. This is
    intentional: MLflow 3.x's `skops` security has been flaky for XGBoost+CalibratedClassifierCV,
    so the joblib is the source of truth and MLflow is the metadata mirror.
    """,
))
