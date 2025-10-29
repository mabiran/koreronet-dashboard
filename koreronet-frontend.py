# app.py (or Visualise.py)
import os, glob, re
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="KōreroNET Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1rem; padding-bottom:1rem;}
    .stTabs [role="tablist"] {gap: 0.5rem;}
    .stTabs [role="tab"] {padding: 0.6rem 1rem; border-radius: 999px; border: 1px solid #3a3a3a;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Hidden configuration (no folder fields in UI)
# ----------------------------
DEFAULT_ROOT = r"G:\My Drive\From the node"
BACKUP_ROOT_DEFAULT = r"G:\My Drive\From the node\Backup"
ROOT = os.environ.get("KORERONET_DATA_ROOT", DEFAULT_ROOT)
BACKUP_ROOT = os.environ.get("KORERONET_BACKUP_ROOT", BACKUP_ROOT_DEFAULT)

# ----------------------------
# CSV discovery
# ----------------------------
@st.cache_data(show_spinner=False)
def list_csvs(root: str):
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    return bn_paths, kn_paths

# Index ALL dates present inside each CSV (robust to DD/MM/YYYY)
@st.cache_data(show_spinner=False)
def extract_dates_from_csv(path: str):
    dates = set()
    try:
        for chunk in pd.read_csv(path, usecols=["ActualTime"], chunksize=5000):
            s = pd.to_datetime(chunk["ActualTime"], errors="coerce", dayfirst=True)
            dates.update(ts.date() for ts in s.dropna())
    except Exception:
        try:
            for chunk in pd.read_csv(path, chunksize=5000):
                if "ActualTime" not in chunk.columns:
                    continue
                s = pd.to_datetime(chunk["ActualTime"], errors="coerce", dayfirst=True)
                dates.update(ts.date() for ts in s.dropna())
        except Exception:
            return []
    return sorted(dates)

@st.cache_data(show_spinner=False)
def build_date_index(bn_paths, kn_paths):
    bn_idx, kn_idx = {}, {}
    for p in bn_paths:
        for d in extract_dates_from_csv(p):
            bn_idx.setdefault(d, []).append(p)
    for p in kn_paths:
        for d in extract_dates_from_csv(p):
            kn_idx.setdefault(d, []).append(p)
    return bn_idx, kn_idx

# ----------------------------
# Normalisation + plotting
# ----------------------------
def standardize_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    df["Confidence"] = pd.to_numeric(df.get("Confidence", np.nan), errors="coerce")

    if kind == "bn":
        if "Common name" in df.columns:
            df["Label"] = df["Common name"]
        else:
            guess = [c for c in df.columns if "common" in c.lower() and "name" in c.lower()]
            df["Label"] = df[guess[0]] if guess else "Unknown"
    else:
        if "Label" not in df.columns:
            possible = [c for c in df.columns if c.lower() in ("label","common name","common_name","species","class")]
            df["Label"] = df[possible[0]] if possible else "Unknown"

    df["ActualTime"] = pd.to_datetime(df.get("ActualTime", pd.NaT), errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Label", "ActualTime"])
    return df[["Label", "Confidence", "ActualTime"]]

def make_heatmap(df: pd.DataFrame, min_conf: float, title: str):
    df_f = df[df["Confidence"].astype(float) >= float(min_conf)].copy()
    if df_f.empty:
        st.warning("No detections after applying the confidence filter.")
        return

    # CHANGED: 12-hour AM/PM axis
    df_f["Hour"] = df_f["ActualTime"].dt.hour
    hour_labels_map = {h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)}
    hour_order = [hour_labels_map[h] for h in range(24)]
    df_f["HourLabel"] = df_f["Hour"].map(hour_labels_map)

    pivot = (df_f.groupby(["Label", "HourLabel"]).size().unstack(fill_value=0).astype(int))

    # Ensure all 24 bins exist and in correct AM/PM order
    for lbl in hour_order:
        if lbl not in pivot.columns:
            pivot[lbl] = 0
    pivot = pivot[hour_order]

    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values(ascending=False).index]

    if pivot.empty:
        st.warning("No data to plot.")
        return

    fig = px.imshow(
        pivot.values,
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="RdYlBu_r",
        labels=dict(x="Hour (AM/PM)", y="Species (common name / label)", color="Detections"),
        text_auto=True,
        aspect="auto",
        title=title,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ----------------------------
# Verify tab helpers
# ----------------------------
def _parse_snapshot(name: str):
    m = re.match(r"^(\d{8})_(\d{6})$", name)
    if not m: return None
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")

@st.cache_data(show_spinner=False)
def list_snapshots(backup_root: str):
    if not os.path.isdir(backup_root): return []
    items = []
    for n in os.listdir(backup_root):
        p = os.path.join(backup_root, n)
        if os.path.isdir(p):
            dt = _parse_snapshot(n)
            if dt: items.append((dt, n, p))
    items.sort(key=lambda x: x[0], reverse=True)
    return items  # [(dt, folder_name, full_path)]

def _read_det_folder(folder: str, kind: str) -> pd.DataFrame:
    frames = []
    for p in glob.glob(os.path.join(folder, "*.csv")):
        try:
            df = pd.read_csv(p, sep=None, engine="python")
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        def col(name, default=np.nan):
            return df[cols[name]] if name in cols else pd.Series([default]*len(df))
        out = pd.DataFrame({
            "chunk_file": col("file", ""),
            "label":      col("label", "Unknown").astype(str),
            "prob":       pd.to_numeric(col("prob"), errors="coerce"),
            "src":        col("src", ""),
            "start_s":    pd.to_numeric(col("start_s"), errors="coerce"),
            "end_s":      pd.to_numeric(col("end_s"), errors="coerce"),
            "det_start":  pd.to_numeric(col("det_start"), errors="coerce") if kind=="bn" else np.nan,
        })
        out["Kind"] = kind.upper()
        out["ChunkPath"] = out["chunk_file"].apply(lambda f: os.path.join(folder, f) if isinstance(f, str) else "")
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["label","prob","src","start_s","end_s","det_start","Kind","ChunkPath"])
    return pd.concat(frames, ignore_index=True)

def _actual_time(src_name: str, offset_seconds: float):
    m = re.match(r"^(\d{8})_(\d{6})", os.path.basename(str(src_name)))
    if not m: return pd.NaT
    base = datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")
    try:
        return base + timedelta(seconds=float(offset_seconds or 0.0))
    except Exception:
        return pd.NaT

def _std_verify(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Label","Confidence","ActualTime","Kind","ChunkPath","Start_s","End_s"])
    use_off = df["det_start"].where(df["det_start"].notna(), df["start_s"])
    times = [_actual_time(s, o) for s, o in zip(df["src"], use_off)]
    out = pd.DataFrame({
        "Label": df["label"].astype(str),
        "Confidence": df["prob"].astype(float),
        "ActualTime": times,
        "Kind": df["Kind"],
        "ChunkPath": df["ChunkPath"],
        "Start_s": df["start_s"],
        "End_s": df["end_s"],
    })
    return out.dropna(subset=["ActualTime"]).sort_values("ActualTime").reset_index(drop=True)

# =========================================
st.title("KōreroNET • Daily Dashboard")
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "📁 Files (soon)"])

# -------------------------
# TAB 1 — calendar + 0.90 default + strict per-day filter
# -------------------------
with tab1:
    bn_paths, kn_paths = list_csvs(ROOT)
    if not bn_paths and not kn_paths:
        st.error("No bn_*.csv / kn_*.csv found.")
        st.stop()

    bn_by_date, kn_by_date = build_date_index(bn_paths, kn_paths)
    bn_dates = sorted(bn_by_date.keys())
    kn_dates = sorted(kn_by_date.keys())
    paired_dates = sorted(set(bn_dates).intersection(set(kn_dates)))

    src = st.selectbox("Source", ["KōreroNET (kn)", "BirdNET (bn)", "Combined"], index=0)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01)

    if src == "Combined":
        options = paired_dates
        help_txt = "Only dates that have BOTH BN & KN detections."
    elif src == "BirdNET (bn)":
        options = bn_dates
        help_txt = "Dates that appear inside any BN file."
    else:
        options = kn_dates
        help_txt = "Dates that appear inside any KN file."

    if not options:
        st.warning(f"No available dates for {src}.")
        st.stop()

    # Calendar instead of selectbox, constrained to available range
    d_default = options[-1]
    d = st.date_input("Day", value=d_default, min_value=options[0], max_value=options[-1], help=help_txt)
    if d not in options:
        earlier = [x for x in options if x <= d]
        d = (earlier[-1] if earlier else options[0])
        st.info(f"No data for the chosen date; showing {d.isoformat()}.")

    def load_and_filter(paths, kind, day_selected):
        frames = []
        for p in paths:
            try:
                df = load_csv(p)
                std = standardize_df(df, kind)
                std = std[std["ActualTime"].dt.date == day_selected]
                if not std.empty:
                    frames.append(std)
            except Exception:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    with st.form(key="det_form"):
        submitted = st.form_submit_button("Update")
        if submitted:
            if src == "BirdNET (bn)":
                df_bn = load_and_filter(bn_by_date.get(d, []), "bn", d)
                make_heatmap(df_bn, min_conf, f"BirdNET • {d.isoformat()}")
            elif src == "KōreroNET (kn)":
                df_kn = load_and_filter(kn_by_date.get(d, []), "kn", d)
                make_heatmap(df_kn, min_conf, f"KōreroNET • {d.isoformat()}")
            else:
                df_bn = load_and_filter(bn_by_date.get(d, []), "bn", d)
                df_kn = load_and_filter(kn_by_date.get(d, []), "kn", d)
                df_all = pd.concat([df_bn, df_kn], ignore_index=True)
                make_heatmap(df_all, min_conf, f"Combined (BN+KN) • {d.isoformat()}")

# -------------------------
# TAB 2 — hidden paths, 0.90 default, calendar filter inside loaded detections
# -------------------------
with tab_verify:
    import base64

    st.subheader("Verify recordings")

    snaps = list_snapshots(BACKUP_ROOT)
    if not snaps:
        st.warning("No snapshot folders like YYYYMMDD_HHMMSS found.")
        st.stop()

    snap_name = st.selectbox("Snapshot", options=[n for _, n, _ in snaps], index=0, key="verify_snap")
    snapshot_path = next(p for _, n, p in snaps if n == snap_name)

    src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0, key="verify_src")
    min_conf_v = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01, key="verify_conf")

    ctx = {"snap": snap_name, "src": src_mode_v, "conf": float(min_conf_v)}
    if st.session_state.get("v_ctx") != ctx:
        st.session_state["v_ctx"] = ctx
        st.session_state["v_loaded"] = False
        st.session_state["v_df"] = pd.DataFrame()

    def _gather_csvs(snapshot_path: str, src_mode: str):
        todo = []
        if src_mode in ("KōreroNET (KN)", "Combined"):
            kn_dir = os.path.join(snapshot_path, "koreronet")
            if os.path.isdir(kn_dir):
                for p in glob.glob(os.path.join(kn_dir, "*.csv")):
                    todo.append(("kn", p))
        if src_mode in ("BirdNET (BN)", "Combined"):
            bn_dir = os.path.join(snapshot_path, "birdnet")
            if os.path.isdir(bn_dir):
                for p in glob.glob(os.path.join(bn_dir, "*.csv")):
                    todo.append(("bn", p))
        if src_mode == "Combined":
            has_kn = any(k == "kn" for k, _ in todo)
            has_bn = any(k == "bn" for k, _ in todo)
            if not (has_kn and has_bn):
                return [], "Combined requires both 'koreronet' and 'birdnet' folders."
        return sorted(todo, key=lambda t: t[1].lower()), None

    def _load_with_progress(csv_list):
        prog = st.progress(0.0)
        status = st.empty()
        frames, n = [], max(1, len(csv_list))
        for i, (kind, path) in enumerate(csv_list, 1):
            status.text(f"Loading {i}/{n}: {os.path.basename(path)}")
            try:
                df = pd.read_csv(path, sep=None, engine="python")
                cols = {c.lower().strip(): c for c in df.columns}
                def col(name, default=np.nan):
                    return df[cols[name]] if name in cols else pd.Series([default]*len(df))
                out = pd.DataFrame({
                    "chunk_file": col("file", ""),
                    "label":      col("label", "Unknown").astype(str),
                    "prob":       pd.to_numeric(col("prob"), errors="coerce"),
                    "src":        col("src", ""),
                    "start_s":    pd.to_numeric(col("start_s"), errors="coerce"),
                    "end_s":      pd.to_numeric(col("end_s"), errors="coerce"),
                    "det_start":  pd.to_numeric(col("det_start"), errors="coerce") if kind=="bn" else np.nan,
                    "Kind":       kind.upper(),
                })
                out["ChunkPath"] = out["chunk_file"].apply(
                    lambda f: os.path.join(os.path.dirname(path), f) if isinstance(f, str) else "")
                frames.append(out)
            except Exception:
                pass
            prog.progress(i/n)
        status.text("Processing…")
        prog.progress(1.0)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # LOAD button
    if st.button("Load detections", type="primary"):
        csv_list, err = _gather_csvs(snapshot_path, src_mode_v)
        if err:
            st.warning(err)
        elif not csv_list:
            st.warning("No CSV files found in the selected snapshot/source.")
        else:
            raw = _load_with_progress(csv_list)
            std_v = _std_verify(raw)
            std_v = std_v[std_v["Confidence"] >= float(min_conf_v)]
            st.session_state["v_df"] = std_v
            st.session_state["v_loaded"] = True
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("v_idx::"):
                    del st.session_state[k]

    if not st.session_state.get("v_loaded", False):
        st.info("Adjust settings and click **Load detections** to enable the player.")
        st.stop()

    std_v = st.session_state.get("v_df", pd.DataFrame())
    if std_v.empty:
        st.warning("No detections after filtering."); st.stop()

    # Calendar filter on Tab 2
    avail_days = sorted(std_v["ActualTime"].dt.date.unique())
    day_default = avail_days[-1]
    day_pick = st.date_input("Day", value=day_default, min_value=avail_days[0], max_value=avail_days[-1],
                             help="Filter the loaded detections by calendar day.")
    if day_pick not in avail_days:
        earlier = [x for x in avail_days if x <= day_pick]
        day_pick = (earlier[-1] if earlier else avail_days[0])
        st.info(f"No detections on that day; showing {day_pick.isoformat()} instead.")
    std_day = std_v[std_v["ActualTime"].dt.date == day_pick]
    if std_day.empty:
        st.warning("No detections for the chosen day."); st.stop()

    # Species list (sorted by count) for the chosen day
    counts = std_day.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox(
        "Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_pick.isoformat()}",
    )

    # Build playlist for selected species & day
    playlist = std_day[std_day["Label"] == species].sort_values("ActualTime").reset_index(drop=True)
    if playlist.empty:
        st.info("No chunks for this species on the chosen day."); st.stop()

    # Index key tied to context (snapshot + source + day + species)
    idx_key = f"v_idx::{snap_name}::{src_mode_v}::{day_pick.isoformat()}::{species}"
    if idx_key not in st.session_state: st.session_state[idx_key] = 0
    idx = st.session_state[idx_key] % len(playlist)

    # Prev / Play / Next (Prev/Next autoplays)
    col1, col2, col3, col4 = st.columns([1,1,1,5])
    autoplay = False
    with col1:
        if st.button("⏮ Prev"):
            idx = (idx - 1) % len(playlist); autoplay = True
    with col2:
        if st.button("▶ Play"):
            autoplay = True
    with col3:
        if st.button("⏭ Next"):
            idx = (idx + 1) % len(playlist); autoplay = True
    st.session_state[idx_key] = idx
    with col4:
        st.markdown(f"**{species}** — {len(playlist)} detections on **{day_pick.isoformat()}** | index {idx+1}/{len(playlist)}")

    row = playlist.iloc[idx]
    meta = (f"**Confidence:** {row['Confidence']:.3f}  |  **Time:** {row['ActualTime']}  |  "
            f"Segment: {float(row['Start_s']):.2f}–{float(row['End_s']):.2f}s")
    st.markdown(meta)

    # Audio player
    def _play_audio(filepath: str, auto: bool):
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            if auto:
                import base64
                b64 = base64.b64encode(data).decode()
                st.markdown(f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>',
                            unsafe_allow_html=True)
            else:
                st.audio(data, format="audio/wav")
        except Exception:
            st.warning("Cannot open audio chunk.")

    _play_audio(row["ChunkPath"], autoplay)

# -------------------------
# TAB 3 (placeholder)
# -------------------------
with tab3:
    st.info("Files tab coming next.")
