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
# Type descriptions (MBTI-like for OTT)
# ================================
TYPE_DESC = {
    "ESFJ": {
        "alias": "ENGAGED",
        "bullets": [
            "앱 활용도 높음·사회성 강함 (트렌드/추천에 민감)",
            "모바일 앱 중심, 주 사용 시간 많음",
            "버라이어티/예능·음악·라이프스타일 선호",
            "한 줄: 재미와 즐거움을 적극적으로 추구"
        ]
    },
    "ESTJ": {
        "alias": "PLANNED",
        "bullets": [
            "자기관리·규율, 시청 시간을 계획적으로 관리",
            "루틴 기반 규칙적 이용, 불필요 플랫폼 정리",
            "뉴스/시사·교양/정보·교육 등 실용 정보 선호",
            "한 줄: 체계적·목적형 OTT 사용"
        ]
    },
    "INTP": {
        "alias": "TARGETED",
        "bullets": [
            "분석적·집중형, 특정 주제에 깊게 몰입",
            "전체 시간은 길지 않아도 선택 시 고밀도 집중",
            "다큐·지식·시리즈 등 심층 콘텐츠 선호",
            "한 줄: 관심 분야만 날카롭게 파고듦"
        ]
    },
    "INFP": {
        "alias": "JOYFUL",
        "bullets": [
            "감성·자유지향, 기분 전환용 즉흥 시청",
            "시간 관리 엄격하진 않음, 스트레스 해소 목적",
            "드라마·로맨스·힐링 예능·음악 선호",
            "한 줄: 즐거움 중심의 자유로운 선택"
        ]
    },
}

# 기본 숫자→문자 매핑(필요 시 사이드바에서 즉석 수정 가능)
DEFAULT_CLUSTER_TO_TYPE = {0: "ESFJ", 1: "ESTJ", 2: "INTP", 3: "INFP"}

def render_type_card(label: str):
    info = TYPE_DESC.get(label)
    if not info:
        st.warning("설명 사전에 없는 유형입니다.")
        return
    st.markdown(
        f"""
        <div style="border:1px solid #eee;border-radius:14px;padding:16px 18px;margin-top:8px;">
          <div style="font-size:1.1rem;font-weight:700;margin-bottom:6px;">
            {label} <span style="opacity:.6;">({info['alias']})</span>
          </div>
          <ul style="margin:0 0 0 1.1rem;">
            {''.join(f'<li style="margin:2px 0;">{b}</li>' for b in info['bullets'])}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

def resolve_label(raw_pred, model, mapping: Dict[int, str]) -> str:
    """
    - 문자열 라벨이면 그대로 반환.
    - 숫자 라벨이면 mapping으로 변환.
    """
    # 문자열 라벨
    if isinstance(raw_pred, str):
        return raw_pred

    # 넘파이 스칼라 → 파이썬 기본형
    if isinstance(raw_pred, (np.generic,)):
        raw_pred = raw_pred.item()

    # 숫자 라벨 -> 매핑
    if isinstance(raw_pred, (int, np.integer)):
        return mapping.get(int(raw_pred), str(raw_pred))

    # 기타 케이스 안전망
    return str(raw_pred)

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
            height: 60vh;  /* 80vh -> 60vh */
            display: flex; align-items: center; justify-content: center; text-align: center;
        }
        .cover-inner h1 { font-size: 3rem; margin-bottom: .25rem; }
        .cover-inner p  { font-size: 1.05rem; color: #555; margin-bottom: 1.25rem; }
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

    # -------- 라벨 매핑(숫자 라벨일 경우) --------
    # 세션 기본 매핑
    if "cluster_to_type" not in st.session_state:
        st.session_state.cluster_to_type = DEFAULT_CLUSTER_TO_TYPE.copy()

    with st.expander("라벨 매핑 (숫자 라벨일 때만 조정하세요)"):
        # classes_가 존재하고 모두 정수면, 그 목록을 보여주고 매핑 수정 허용
        classes = getattr(model, "classes_", None)
        if classes is not None and all(isinstance(c, (int, np.integer)) for c in classes):
            for c in sorted([int(x) for x in classes]):
                st.session_state.cluster_to_type[c] = st.selectbox(
                    f"클러스터 {c} → 유형",
                    options=list(TYPE_DESC.keys()),
                    index=list(TYPE_DESC.keys()).index(
                        st.session_state.cluster_to_type.get(c, DEFAULT_CLUSTER_TO_TYPE.get(c, "ESFJ"))
                    ),
                    key=f"map_{c}"
                )
        else:
            st.caption("모델이 문자열 라벨을 직접 반환하면 매핑은 필요하지 않습니다.")

# ================================
# Friendly input widgets
# ================================
def time_hours_widget(label: str, key: str, minute_mode: bool, max_h: int = 70) -> float:
    """
    주당 '시간' 입력.
    - minute_mode=False: 0.25h(15분) 단위 슬라이더
    - minute_mode=True : 시/분 분리 입력 → 시간(float)로 변환
    """
    if not minute_mode:
        return st.slider(label, min_value=0.0, max_value=float(max_h),
                         value=0.0, step=0.25, key=key,
                         help="15분=0.25h, 30분=0.5h, 1시간=1.0h")
    else:
        c_h, c_m = st.columns([2,1])
        with c_h:
            hh = st.number_input(f"{label} (시간)", min_value=0, max_value=max_h, value=0, step=1, key=f"{key}_h")
        with c_m:
            mm = st.number_input(f"{label} (분)",   min_value=0, max_value=59, value=0, step=5, key=f"{key}_m")
        return float(hh) + float(mm)/60.0

def count_per_week_widget(label: str, key: str, max_cnt: int = 70) -> int:
    """주당 횟수 입력(정수)."""
    return st.number_input(label, min_value=0, max_value=max_cnt, value=0, step=1, format="%d", key=key)

# ================================
# 입력 영역
# ================================
st.markdown("### 이용 패턴 입력")
minute_mode = st.toggle("시/분으로 입력할래요? (끄면 15분 단위 슬라이더)", value=False)

c1, c2, c3 = st.columns(3)
with c1:
    major_ott = time_hours_widget("Major OTT (주당 시청시간, 시간)", key="major", minute_mode=minute_mode)
    youtube   = time_hours_widget("YouTube (주당 시청시간, 시간)",   key="yt",    minute_mode=minute_mode)
with c2:
    minor_ott = time_hours_widget("Minor OTT (주당 시청시간, 시간)", key="minor", minute_mode=minute_mode)
    shopping  = count_per_week_widget("쇼핑 (주당 이용 횟수, 회)",     key="shop")
with c3:
    sports    = time_hours_widget("스포츠 (주당 시청시간, 시간)",     key="sports", minute_mode=minute_mode)
    media_ott_val = st.selectbox("미디어_OTT (사용 OTT 수)", options=list(range(0, 11)), index=0,
                                 help="동시에 사용하는 OTT 서비스의 개수")

st.caption("※ 시청시간은 '시간' 단위로 모델에 들어갑니다. (예: 1시간 30분 → 1.5시간)")

# ================================
# TV 장르(X1,X2,X3) + 동영상 장르 체크
# ================================
st.markdown("### TV 장르 순위 선택 (X1, X2, X3)")
colx1, colx2, colx3 = st.columns(3)
genre_options = list(GENRE_MAP.values())
with colx1:
    x1_label = st.selectbox("1순위 장르", options=genre_options, index=0)
with colx2:
    x2_label = st.selectbox("2순위 장르", options=genre_options, index=0)
with colx3:
    x3_label = st.selectbox("3순위 장르", options=genre_options, index=0)

st.markdown("### 동영상 콘텐츠 장르 (해당 시 체크)")
x_onoff_cols = [c for c in FEATURE_COLS if c.startswith("X") and c[1:].isdigit() and int(c[1:]) >= 6]
onoff_selections: Dict[str, int] = {}
cols = st.columns(3)
for i, colname in enumerate(sorted(x_onoff_cols, key=lambda s: int(s[1:]))):
    label = X_LABELS.get(colname, colname)
    with cols[i % 3]:
        onoff_selections[colname] = st.checkbox(label, value=False)

# ================================
# 예측 실행
# ================================
if st.button("예측 실행", type="primary"):
    base_nums = {
        "Major OTT": major_ott,        # 시간(h)
        "Minor OTT": minor_ott,        # 시간(h)
        "YouTube": youtube,            # 시간(h)
        "스포츠": sports,               # 시간(h)
        "쇼핑": float(shopping),        # 횟수
        "미디어_OTT": float(media_ott_val),
    }
    x123_vals = {
        "X1": GENRE_LABEL_TO_CODE.get(x1_label),
        "X2": GENRE_LABEL_TO_CODE.get(x2_label),
        "X3": GENRE_LABEL_TO_CODE.get(x3_label),
    }

    Xrow = build_manual_row(FEATURE_COLS, base_nums, x123_vals, onoff_selections)
    raw_pred = model.predict(Xrow.to_numpy())[0]

    # 숫자/문자 라벨 모두 처리
    pred_label = resolve_label(raw_pred, model, st.session_state.cluster_to_type)

    st.success(f"예측 군집: **{pred_label}**")
    render_type_card(pred_label)

    # (옵션) 확률 막대
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(Xrow.to_numpy())[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                # 각 클래스 라벨을 화면용 문자열로 정규화
                view_labels = []
                for c in classes:
                    view_labels.append(resolve_label(c, model, st.session_state.cluster_to_type))
                dfp = pd.DataFrame({"class": view_labels, "prob": probs}).sort_values("prob", ascending=False)
                st.bar_chart(dfp.set_index("class"))
        except Exception:
            pass

