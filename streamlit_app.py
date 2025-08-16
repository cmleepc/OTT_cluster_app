import os
import pickle
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================================
# Paths
# ================================
APP_DIR = Path(__file__).parent.resolve()
FIRST_CSV   = APP_DIR / "first.csv"
SECOND_CSV  = APP_DIR / "second.csv"
MODEL_PKL   = APP_DIR / "model.pkl"

# ================================
# OTT 그룹 설명 (툴팁에 사용)
# ================================
MAJOR_APPS = [
    "Disney+ (디즈니+)",
    "쿠팡플레이",
    "Wavve(웨이브)",
    "TVING",
    "Netflix(넷플릭스)",
]
MINOR_APPS = [
    "아프리카TV (AfreecaTV)",
    "Twitch: 게임 생방송",
    "U+모바일tv",
    "왓챠",
    "SBS - 온에어/VOD/방청",
    "KBS+",
    "NAVER NOW",
    "MBC",
    "네이버 시리즈온 (SERIES ON)",
]
MAJOR_HELP = "메이저 OTT 예시:\n- " + "\n- ".join(MAJOR_APPS)
MINOR_HELP = "마이너 OTT 예시:\n- " + "\n- ".join(MINOR_APPS)

# ================================
# Genre map
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
# MBTI 별칭/설명/매핑
# ================================
TYPE_ALIAS = {"ESFJ": "ENGAGED", "ESTJ": "PLANNED", "INTP": "TARGETED", "INFP": "JOYFUL"}
ALIAS_TO_TYPE = {
    "ENGAGED": "ESFJ", "STIMULATING": "ESFJ", "FRAGMENTED": "ESFJ",
    "PLANNED": "ESTJ", "NECESSITYFOCUSED": "ESTJ", "NECCESITYFOCUSED": "ESTJ",
    "TARGETED": "INTP",
    "JOYFUL": "INFP", "IDLE": "INFP",
    "ESFJ": "ESFJ", "ESTJ": "ESTJ", "INTP": "INTP", "INFP": "INFP",
}
DEFAULT_CLUSTER_TO_TYPE = {0: "ESFJ", 1: "ESTJ", 2: "INTP", 3: "INFP"}

DIM_DESC = {
    "E": "외향(E): OTT 사용량이 많고 다양한 앱을 적극적으로 활용합니다.",
    "I": "내향(I): OTT 사용량이 적고 혼자 보는 선택적·조용한 이용을 선호합니다.",
    "S": "감각(S): 모바일 앱 중심으로 일상 속에 자연스럽게 OTT를 사용합니다.",
    "N": "직관(N): 엔터테인먼트/미디어 방식으로 호기심에 따라 폭넓게 탐색합니다.",
    "T": "사고(T): 실용성과 필요성을 중시하며 효율적으로 콘텐츠를 고릅니다.",
    "F": "감정(F): 즐거움·공감을 중시하며 감정적 만족을 위해 시청합니다.",
    "J": "판단(J): 자기관리와 계획을 세워 시청 패턴을 꾸준히 유지합니다.",
    "P": "인식(P): 자유·즉흥적으로 상황에 따라 유연하게 시청합니다.",
}
SUMMARY_LINE = {
    "ESFJ": "외향(E)+감각(S)+감정(F)+계획형(J) 조합으로, 많이 즐기되 질서 있게 사용하는 타입입니다.",
    "ESTJ": "외향(E)+감각(S)+사고(T)+계획형(J) 조합으로, 목적과 효율 중심의 체계적 사용자입니다.",
    "INTP": "내향(I)+직관(N)+사고(T)+인식형(P) 조합으로, 적은 양을 선택·집중해 깊게 파는 탐구형 사용자입니다.",
    "INFP": "내향(I)+직관(N)+감정(F)+인식형(P) 조합으로, 감정 이입과 휴식을 위해 자유롭게 시청하는 사용자입니다.",
}

# ---------- 공통 유틸 ----------
def _norm_str(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())

def mbti_letters(label: str) -> str:
    s = "".join(ch for ch in str(label).upper() if ch.isalpha())
    if len(s) >= 4:
        four = s[:4]
        ok = (four[0] in "EI") and (four[1] in "SN") and (four[2] in "TF") and (four[3] in "JP")
        return four if ok else s[:4]
    return s

def resolve_to_mbti(raw_pred, cluster_map: Dict[int, str]) -> str:
    if isinstance(raw_pred, (np.generic,)): raw_pred = raw_pred.item()
    if isinstance(raw_pred, str):
        key = _norm_str(raw_pred)
        if key in ALIAS_TO_TYPE:
            return ALIAS_TO_TYPE[key]
        if key.isdigit():
            return cluster_map.get(int(key), str(raw_pred))
        return str(raw_pred)
    if isinstance(raw_pred, (int, np.integer)):
        return cluster_map.get(int(raw_pred), str(raw_pred))
    return str(raw_pred)

def aggregate_probs_by_type(classes, probs, cluster_map: Dict[int, str]) -> pd.DataFrame:
    """
    모델 클래스 확률을 ESFJ/ESTJ/INTP/INFP로 모아 집계하고,
    집계합으로 정규화(합계 1.0)하여 반환.
    """
    label_probs: Dict[str, float] = {"ESFJ":0.0, "ESTJ":0.0, "INTP":0.0, "INFP":0.0}

    for c, p in zip(classes, probs):
        mapped = resolve_to_mbti(c, cluster_map)
        if mapped in label_probs:
            label_probs[mapped] += float(p)
        # 매핑되지 않는 클래스는 무시(= 집계합에서 자동 제외)

    total = sum(label_probs.values())
    if total > 0:
        for k in label_probs:
            label_probs[k] /= total

    dfp = pd.DataFrame({"class": list(label_probs.keys()), "prob": list(label_probs.values())})
    return dfp.sort_values("prob", ascending=False)

def render_combined_profile(label: str):
    mbti = mbti_letters(label)
    alias = TYPE_ALIAS.get(mbti, "")
    bullets = [DIM_DESC[ch] for ch in mbti if ch in DIM_DESC]
    summary = SUMMARY_LINE.get(mbti, "")
    st.markdown(
        f"""
        <div style="border:1px solid #eee;border-radius:14px;padding:16px 18px;margin:8px 0;">
          <div style="font-size:1.1rem;font-weight:700;margin-bottom:6px;">
            {mbti} {f"({alias})" if alias else ""}
          </div>
          <ul style="margin:0 0 0 1.1rem;">
            {''.join(f'<li style="margin:2px 0;">{b}</li>' for b in bullets)}
          </ul>
          <div style="margin-top:8px;"><b>요약:</b> {summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- 막대 라벨: 항상 막대 위(검정) + 상단 여유 ----
def plot_probs_with_labels(prob_df: pd.DataFrame):
    if prob_df is None or prob_df.empty:
        return
    labels = prob_df["class"].tolist()
    vals   = prob_df["prob"].tolist()
    perc   = [v * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(labels, vals)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Probability")

    for i, b in enumerate(bars):
        h = b.get_height()
        y = min(h + 0.03, 1.05)
        ax.text(b.get_x() + b.get_width()/2, y, f"{perc[i]:.1f}%",
                ha="center", va="bottom", color="black", fontweight="bold", fontsize=11)

    st.pyplot(fig, clear_figure=True)

# ================================
# Data & Preprocess
# ================================
@st.cache_data(show_spinner=False)
def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(FIRST_CSV)
    df2 = pd.read_csv(SECOND_CSV)  # row0 has labels for X6+
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
# 입력행 생성 유틸
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
        if k in row.columns: row.at[0, k] = float(v)
    for xk, val in x123_values.items():
        if val is None: continue
        col_name = f"{xk}_{val}"
        if col_name in row.columns: row.at[0, col_name] = 1.0
    for xk, onoff in onoff_map.items():
        if xk in row.columns: row.at[0, xk] = int(onoff)
    return row

# ================================
# UI
# ================================
st.set_page_config(page_title="OTT 이용자 군집 예측", layout="wide")

# session state
if "started" not in st.session_state:
    st.session_state.started = False
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
if "modal_token" not in st.session_state:
    st.session_state.modal_token = 0
if "modal_last_shown" not in st.session_state:
    st.session_state.modal_last_shown = -1

# ---- 커버 ----
if not st.session_state.started:
    st.markdown(
        """
        <style>
        .cover-wrap { height: 45vh; display:flex; align-items:center; justify-content:center; text-align:center; }
        .cover-inner h1 { font-size:3rem; margin-bottom:.25rem; }
        .cover-inner p  { font-size:1.05rem; color:#555; margin-bottom:1rem; }
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

st.title("📺 OTT 이용자 군집 예측")

with st.sidebar:
    st.header("설정")
    FEATURE_COLS, X_LABELS = prepare_training_schema()
    model = load_or_train_model(FEATURE_COLS)
    st.success("스키마 & 모델 준비 완료 ✅")

    if "cluster_to_type" not in st.session_state:
        st.session_state.cluster_to_type = DEFAULT_CLUSTER_TO_TYPE.copy()

    with st.expander("라벨 매핑 (숫자/별칭일 때 조정)"):
        classes = getattr(model, "classes_", None)
        if classes is not None:
            st.caption(f"model.classes_: {list(classes)}")
        for k in sorted(st.session_state.cluster_to_type.keys()):
            st.session_state.cluster_to_type[k] = st.selectbox(
                f"클러스터 {k} → 유형",
                options=list(TYPE_ALIAS.keys()),
                index=list(TYPE_ALIAS.keys()).index(st.session_state.cluster_to_type[k]),
                key=f"map_{k}"
            )

# ================================
# 입력부: (시간)(분) 쌍 – 동일 너비, 메이저/마이너 툴팁
# ================================
st.markdown("### 이용 패턴 입력")

def time_pair_in_columns(col_h, col_m, title: str, key: str, max_h: int = 72, help_text: str | None = None) -> float:
    with col_h:
        hh = st.number_input(f"{title} (시간)", min_value=0, max_value=max_h, value=0, step=1,
                             key=f"{key}_h", help=help_text)
    with col_m:
        mm = st.number_input("(분)", min_value=0, max_value=59, value=0, step=5, key=f"{key}_m")
    return float(hh) + float(mm)/60.0

# 1행: Major OTT | Minor OTT  (각 라벨에 예시 툴팁)
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
major_ott = time_pair_in_columns(r1c1, r1c2, "Major OTT", "major", help_text=MAJOR_HELP)
minor_ott = time_pair_in_columns(r1c3, r1c4, "Minor OTT", "minor", help_text=MINOR_HELP)

# 2행: YouTube | 스포츠
r2c1, r2c2, r2c3, r2c4 = st.columns(4)
youtube = time_pair_in_columns(r2c1, r2c2, "YouTube", "yt")
sports  = time_pair_in_columns(r2c3, r2c4, "스포츠", "sports")

# 3행: 쇼핑 / 사용 OTT 수
r3c1, r3c2 = st.columns(2)
with r3c1:
    shopping = st.number_input("쇼핑 (주당 이용 횟수, 회)",
                               min_value=0, max_value=70, value=0, step=1, format="%d", key="shop")
with r3c2:
    media_ott_val = st.selectbox("사용 OTT 수 (개)", options=list(range(0, 11)), index=0,
                                 help="동시에 사용하는 OTT 서비스 개수")

# ================================
# TV 장르(X1~X3) + 동영상 장르 체크
# ================================
st.markdown("### 선호 TV 장르 선택")
colx1, colx2, colx3 = st.columns(3)
genre_options = list(GENRE_MAP.values())
with colx1:
    x1_label = st.selectbox("1순위 장르", options=genre_options, index=0)
with colx2:
    x2_label = st.selectbox("2순위 장르", options=genre_options, index=0)
with colx3:
    x3_label = st.selectbox("3순위 장르", options=genre_options, index=0)

st.markdown("### 선호 동영상 콘텐츠 장르 (중복체크)")
x_onoff_cols = [c for c in FEATURE_COLS if c.startswith("X") and c[1:].isdigit() and int(c[1:]) >= 6]
onoff_selections: Dict[str, int] = {}
cols = st.columns(3)
for i, colname in enumerate(sorted(x_onoff_cols, key=lambda s: int(s[1:]))):
    label = X_LABELS.get(colname, colname)
    with cols[i % 3]:
        val = st.checkbox(label, value=False, key=f"on_{colname}")
        onoff_selections[colname] = val

# ================================
# 결과 모달(dialog)
# ================================
def _result_body(pred_label: str, prob_df: pd.DataFrame | None):
    st.success(f"예측 군집: **{pred_label}**")
    render_combined_profile(pred_label)
    if prob_df is not None and not prob_df.empty:
        st.markdown("---")
        st.caption("클래스 확률(4유형 집계)")
        plot_probs_with_labels(prob_df)

HAS_DIALOG = hasattr(st, "dialog")

if HAS_DIALOG:
    @st.dialog("예측 결과", width="large")
    def show_result_dialog():
        pred_label = st.session_state.get("result_label")
        prob_df = st.session_state.get("result_probs")
        _result_body(pred_label, prob_df)
        st.divider()
        if st.button("닫기", use_container_width=True):
            st.session_state.show_modal = False
            st.rerun()
else:
    def show_result_dialog():
        st.markdown("""
        <style>
        .overlay { position:fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,.35); z-index:1000; }
        .modal { position:fixed; top: 8vh; left:50%; transform:translateX(-50%);
                 width:min(860px,94vw); background:#fff; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.2);
                 padding:18px 20px; z-index:1001; }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="overlay"></div><div class="modal">', unsafe_allow_html=True)
        pred_label = st.session_state.get("result_label")
        prob_df = st.session_state.get("result_probs")
        _result_body(pred_label, prob_df)
        if st.button("닫기", use_container_width=True):
            st.session_state.show_modal = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ================================
# 예측 실행
# ================================
if st.button("예측 실행", type="primary"):
    base_nums = {
        "Major OTT": major_ott,
        "Minor OTT": minor_ott,
        "YouTube": youtube,
        "스포츠": sports,
        "쇼핑": float(shopping),
        "미디어_OTT": float(media_ott_val),   # 내부 컬럼명은 기존 유지
    }
    x123_vals = {
        "X1": GENRE_LABEL_TO_CODE.get(x1_label),
        "X2": GENRE_LABEL_TO_CODE.get(x2_label),
        "X3": GENRE_LABEL_TO_CODE.get(x3_label),
    }

    Xrow = build_manual_row(FEATURE_COLS, base_nums, x123_vals, onoff_selections)
    raw_pred = model.predict(Xrow.to_numpy())[0]
    pred_label = resolve_to_mbti(raw_pred, st.session_state.cluster_to_type)

    prob_df = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(Xrow.to_numpy())[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                prob_df = aggregate_probs_by_type(classes, probs, st.session_state.cluster_to_type)
        except Exception:
            pass

    st.session_state.result_label = pred_label
    st.session_state.result_probs = prob_df
    st.session_state.show_modal = True
    st.session_state.modal_token += 1  # 이번 실행 토큰 갱신

# 토큰 방식: 같은 토큰에서는 1번만 모달 표시
if st.session_state.show_modal:
    token = st.session_state.modal_token
    if st.session_state.modal_last_shown != token:
        show_result_dialog()
        st.session_state.modal_last_shown = token







