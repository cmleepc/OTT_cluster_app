
import os
import pickle
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ================================
# Paths
# ================================
APP_DIR = Path(__file__).parent.resolve()
FIRST_CSV   = APP_DIR / "first.csv"
SECOND_CSV  = APP_DIR / "second.csv"
MODEL_PKL   = APP_DIR / "model.pkl"

# ================================
# Genre map (hardcoded from user)
# ================================
GENRE_MAP = {
    1: "버라이어티/예능",
    2: "드라마",
    3: "뉴스",
    4: "스포츠",
    5: "취미/레저",
    6: "음악",
    7: "교육",
    8: "시사/다큐",
    9: "교양/정보",
    10: "홈쇼핑",
    11: "성인",
    997: "기타"
}
GENRE_LABEL_TO_CODE = {v: k for k, v in GENRE_MAP.items()}

# ================================
# Data & Preprocess (mirrors modeling.py decisions)
# ================================
@st.cache_data(show_spinner=False)
def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(FIRST_CSV)   # first.csv
    df2 = pd.read_csv(SECOND_CSV)  # second.csv (row0 has labels for X6+)
    return df1, df2

def _rename_second_like_training(df2: pd.DataFrame) -> pd.DataFrame:
    df2 = df2.copy()
    df2 = df2.rename(columns={df2.columns[0]: "panel_id"})
    rename_map = {old: f"X{i+1}" for i, old in enumerate(df2.columns[1:])}
    df2 = df2.rename(columns=rename_map)
    return df2

def build_human_label_map(df2_raw: pd.DataFrame) -> Dict[str, str]:
    labels_row = df2_raw.iloc[0]
    colnames   = list(df2_raw.columns)
    friendly: Dict[str, str] = {}
    for idx, col in enumerate(colnames[1:], start=1):
        if idx >= 6:
            friendly[f"X{idx}"] = str(labels_row[col])
    return friendly

@st.cache_data(show_spinner=False)
def prepare_training_schema() -> Tuple[List[str], Dict[str, str]]:
    df1_raw, df2_raw = load_raw()
    allowed_ids = set(df1_raw['panel_id'].astype(str))
    df2 = df2_raw[df2_raw.iloc[:, 0].astype(str).isin(allowed_ids)].reset_index(drop=True)

    x_label_map = build_human_label_map(df2_raw)

    df2 = _rename_second_like_training(df2)
    for cat in ["X1", "X2", "X3"]:
        if cat in df2.columns:
            df2[cat] = pd.to_numeric(df2[cat], errors="coerce")
    df2 = pd.get_dummies(df2, columns=["X1", "X2", "X3"], drop_first=True, dtype=int)

    merged = pd.merge(df1_raw, df2, on="panel_id", how="inner")
    feature_cols = [c for c in merged.columns if c not in ["panel_id", "cluster"]]
    return feature_cols, x_label_map

# ================================
# Model loader (with fallback)
# ================================
@st.cache_resource(show_spinner=False)
def load_or_train_model(feature_cols: List[str]):
    model = None
    try:
        with open(MODEL_PKL, "rb") as f:
            model = pickle.load(f)
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        df1_raw, df2_raw = load_raw()
        allowed_ids = set(df1_raw['panel_id'].astype(str))
        df2 = df2_raw[df2_raw.iloc[:, 0].astype(str).isin(allowed_ids)].reset_index(drop=True)
        df2 = _rename_second_like_training(df2)
        for cat in ["X1", "X2", "X3"]:
            if cat in df2.columns:
                df2[cat] = pd.to_numeric(df2[cat], errors="coerce")
        df2 = pd.get_dummies(df2, columns=["X1", "X2", "X3"], drop_first=True, dtype=int)
        merged = pd.merge(df1_raw, df2, on="panel_id", how="inner")
        X = merged[feature_cols].to_numpy()
        y = merged["cluster"].to_numpy()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(X, y)
        model = rf
    return model

# ================================
# Utilities
# ================================
def empty_feature_row(feature_cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame([np.zeros(len(feature_cols), dtype=float)], columns=feature_cols)

def build_manual_row(
    feature_cols: List[str],
    base_nums: Dict[str, float],
    x123_values: Dict[str, int],
    onoff_map: Dict[str, int]
) -> pd.DataFrame:
    row = empty_feature_row(feature_cols)

    for k, v in base_nums.items():
        if k in row.columns:
            row.at[0, k] = float(v)

    for xk, val in x123_values.items():
        if val is None:
            continue
        col_name = f"{xk}_{val}"
        if col_name in row.columns:
            row.at[0, col_name] = 1.0

    for xk, onoff in onoff_map.items():
        if xk in row.columns:
            row.at[0, xk] = int(onoff)

    return row

# ================================
# UI
# ================================
st.set_page_config(page_title="OTT 이용자 군집 예측", layout="wide")

# ---- Cover page (centered) ----
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown(
        """
        <style>
        .cover-wrap {
            height: 80vh; display: flex; align-items: center; justify-content: center; text-align: center;
        }
        .cover-inner h1 { font-size: 3rem; margin-bottom: .5rem; }
        .cover-inner p  { font-size: 1.1rem; color: #555; margin-bottom: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="cover-wrap">
          <div class="cover-inner">
            <h1>📺 OTT 이용자 군집 예측</h1>
            <p>이용 패턴과 선호 장르를 입력하면 군집을 예측합니다.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    start_col = st.columns([1,1,1])[1]
    with start_col:
        if st.button("시작하기", type="primary", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    st.stop()

# ---- Prediction page ----
st.title("📺 OTT 이용자 군집 예측")

with st.sidebar:
    st.header("설정")
    FEATURE_COLS, X_LABELS = prepare_training_schema()
    model = load_or_train_model(FEATURE_COLS)
    st.success("스키마 & 모델 준비 완료 ✅")

# 입력 영역
c1, c2, c3 = st.columns(3)
with c1:
    major_ott = st.number_input("Major OTT", min_value=0.0, step=0.1, value=0.0)
    youtube   = st.number_input("YouTube", min_value=0.0, step=0.1, value=0.0)
with c2:
    minor_ott = st.number_input("Minor OTT", min_value=0.0, step=0.1, value=0.0)
    shopping  = st.number_input("쇼핑", min_value=0.0, step=0.1, value=0.0)
with c3:
    sports    = st.number_input("스포츠", min_value=0.0, step=0.1, value=0.0)
    # ▶ 항상 노출: 미디어_OTT (사용 OTT 수)
    media_ott_val = st.selectbox("미디어_OTT (사용 OTT 수)", options=list(range(0, 11)), index=0)

st.markdown("### TV 장르 순위 선택 (X1, X2, X3)")
colx1, colx2, colx3 = st.columns(3)
genre_options = list(GENRE_MAP.values())
with colx1:
    x1_label = st.selectbox("1순위 장르", options=genre_options, index=0)
with colx2:
    x2_label = st.selectbox("2순위 장르", options=genre_options, index=0)
with colx3:
    x3_label = st.selectbox("3순위 장르", options=genre_options, index=0)

# X6..Xn on/off 체크박스
st.markdown("### 동영상 콘텐츠 장르 (해당 시 체크)")
x_onoff_cols = [c for c in FEATURE_COLS if c.startswith("X") and c[1:].isdigit() and int(c[1:]) >= 6]
onoff_selections: Dict[str, int] = {}
cols = st.columns(3)
for i, colname in enumerate(sorted(x_onoff_cols, key=lambda s: int(s[1:]))):
    label = X_LABELS.get(colname, colname)
    with cols[i % 3]:
        onoff_selections[colname] = st.checkbox(label, value=False)

# 예측 실행
if st.button("예측 실행", type="primary"):
    base_nums = {
        "Major OTT": major_ott,
        "Minor OTT": minor_ott,
        "YouTube": youtube,
        "쇼핑": shopping,
        "스포츠": sports,
        "미디어_OTT": float(media_ott_val),   # ← 항상 세팅 (모델 피처에 없으면 자동 무시됨)
    }
    x123_vals = {
        "X1": GENRE_LABEL_TO_CODE.get(x1_label),
        "X2": GENRE_LABEL_TO_CODE.get(x2_label),
        "X3": GENRE_LABEL_TO_CODE.get(x3_label),
    }

    Xrow = build_manual_row(FEATURE_COLS, base_nums, x123_vals, onoff_selections)
    pred = model.predict(Xrow.to_numpy())[0]
    st.success(f"예측 군집: **{pred}**")

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(Xrow.to_numpy())[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                dfp = pd.DataFrame({"class": classes, "prob": probs}).sort_values("prob", ascending=False)
                st.bar_chart(dfp.set_index("class"))
        except Exception:
            pass
