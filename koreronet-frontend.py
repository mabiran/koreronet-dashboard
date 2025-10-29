# app.py (or Visualise.py) — with Drive Diagnostics

import os, glob, re, json, io
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Google Drive config (from Streamlit Secrets)
# ---------------------------------------------------------
GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID")
SERVICE_ACCOUNT_JSON = st.secrets.get("SERVICE_ACCOUNT_JSON")

CACHE_ROOT = Path("/tmp/gdrive_cache")   # ephemeral on Streamlit Cloud
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Will fill these in below
drive = None
gdrive_enabled = False
gdrive_err = None
sa_email = None
sa_project = None
sa_key_id = None

# ---------------------------------------------------------
# Attempt auth/build Drive client
# ---------------------------------------------------------
if GDRIVE_FOLDER_ID and SERVICE_ACCOUNT_JSON:
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError

        # Secrets may be a dict (Streamlit TOML parsed it) or a str
        sa_info = SERVICE_ACCOUNT_JSON
        if isinstance(sa_info, str):
            sa_info = json.loads(sa_info)

        sa_email = sa_info.get("client_email")
        sa_project = sa_info.get("project_id")
        sa_key_id = sa_info.get("private_key_id")

        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )

        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        gdrive_enabled = True
    except Exception as e:
        gdrive_err = f"{type(e).__name__}: {e}"
else:
    if not GDRIVE_FOLDER_ID:
        gdrive_err = "Missing GDRIVE_FOLDER_ID in Streamlit secrets."
    elif not SERVICE_ACCOUNT_JSON:
        gdrive_err = "Missing SERVICE_ACCOUNT_JSON in Streamlit secrets."

# ---------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------
if gdrive_enabled:
    @st.cache_data(show_spinner=False, ttl=300)
    def list_children(folder_id: str):
        """List direct children of a folder (supports My Drive & Shared Drives)."""
        items, token = [], None
        fields = "nextPageToken, files(id, name, mimeType, modifiedTime, md5Checksum, parents, driveId)"
        q = f"'{folder_id}' in parents and trashed=false"
        while True:
            resp = drive.files().list(
                q=q,
                fields=fields,
                pageToken=token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            items.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return items

    @st.cache_data(show_spinner=False, ttl=300)
    def list_tree_recursive(root_folder_id: str, max_nodes: int = 2000):
        """BFS over folders to build a (path, item) list; capped to avoid huge listings."""
        from collections import deque
        out = []
        q = deque([("", root_folder_id)])  # (path_prefix, folder_id)
        visited = set()
        nodes = 0
        while q and nodes < max_nodes:
            prefix, fid = q.popleft()
            if fid in visited:
                continue
            visited.add(fid)
            try:
                for it in list_children(fid):
                    nodes += 1
                    name = it["name"]
                    is_folder = it.get("mimeType", "").endswith("folder")
                    path = f"{prefix}/{name}".lstrip("/")
                    out.append((path, it))
                    if is_folder:
                        q.append((path, it["id"]))
                    if nodes >= max_nodes:
                        break
            except HttpError as e:
                out.append((f"{prefix}/<error: {fid}>", {"error": str(e)}))
        return out

    def _download_file(file_id: str, local_path: Path):
        req = drive.files().get_media(fileId=file_id)
        with open(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def sync_csvs_recursive(root_folder_id: str, patterns=("bn", "kn")) -> int:
        """Download bn*.csv / kn*.csv found anywhere under the root folder."""
        items = list_tree_recursive(root_folder_id)
        targets = [
            it for (p, it) in items
            if it.get("mimeType", "") != "application/vnd.google-apps.folder"
            and it["name"].lower().endswith(".csv")
            and it["name"].lower().startswith(tuple(p.lower() for p in patterns))
        ]
        count = 0
        for it in targets:
            local_path = CACHE_ROOT / it["name"]
            if local_path.exists():
                continue
            _download_file(it["id"], local_path)
            count += 1
        return count

# ---------------------------------------------------------
# Choose data roots (Drive cache if available)
# ---------------------------------------------------------
if gdrive_enabled:
    try:
        _ = sync_csvs_recursive(GDRIVE_FOLDER_ID)
    except Exception as e:
        gdrive_err = f"Sync error: {type(e).__name__}: {e}"
        gdrive_enabled = False

if gdrive_enabled:
    ROOT = str(CACHE_ROOT)
    BACKUP_ROOT = str(CACHE_ROOT / "Backup")
else:
    DEFAULT_ROOT = r"G:\My Drive\From the node"
    BACKUP_ROOT_DEFAULT = r"G:\My Drive\From the node\Backup"
    ROOT = os.environ.get("KORERONET_DATA_ROOT", DEFAULT_ROOT)
    BACKUP_ROOT = os.environ.get("KORERONET_BACKUP_ROOT", BACKUP_ROOT_DEFAULT)

# Status banner
st.info(
    f"Data source: {'Google Drive cache (/tmp/gdrive_cache)' if gdrive_enabled else ROOT}"
    + (f"  •  Note: {gdrive_err}" if gdrive_err else "")
)

# ---------------------------------------------------------
# CSV discovery / parsing
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_csvs(root: str):
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    return bn_paths, kn_paths

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

# ---------------------------------------------------------
# Normalisation + plotting
# ---------------------------------------------------------
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

    df_f["Hour"] = df_f["ActualTime"].dt.hour
    hour_labels_map = {h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)}
    hour_order = [hour_labels_map[h] for h in range(24)]
    df_f["HourLabel"] = df_f["Hour"].map(hour_labels_map)
    pivot = (df_f.groupby(["Label", "HourLabel"]).size().unstack(fill_value=0).astype(int))

    for lbl in hour_order:
        if lbl not in pivot.columns:
            pivot[lbl] = 0
    pivot = pivot[hour_order]

    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values(ascending=False).index]

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

# ---------------------------------------------------------
# Verify tab helpers
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
st.title("KōreroNET • Daily Dashboard")
tab1, tab_verify, tab_diag = st.tabs(["📊 Detections", "🎧 Verify recordings", "🧪 Drive Diagnostics"])

# -------------------------
# TAB 1 — detections
# -------------------------
with tab1:
    bn_paths, kn_paths = list_csvs(ROOT)
    if not bn_paths and not kn_paths:
        st.error("No bn_*.csv / kn_*.csv found in: " + ROOT)
        if gdrive_enabled:
            st.caption("Tip: ensure the service account has access and that CSVs start with bn/kn.")
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
# TAB 2 — verify recordings
# -------------------------
with tab_verify:
    import base64

    st.subheader("Verify recordings")

    snaps = list_snapshots(BACKUP_ROOT)
    if not snaps:
        st.warning("No snapshot folders like YYYYMMDD_HHMMSS found under: " + BACKUP_ROOT)
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

    counts = std_day.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox(
        "Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_pick.isoformat()}",
    )

    playlist = std_day[std_day["Label"] == species].sort_values("ActualTime").reset_index(drop=True)
    if playlist.empty:
        st.info("No chunks for this species on the chosen day."); st.stop()

    idx_key = f"v_idx::{snap_name}::{src_mode_v}::{day_pick.isoformat()}::{species}"
    if idx_key not in st.session_state: st.session_state[idx_key] = 0
    idx = st.session_state[idx_key] % len(playlist)

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
# TAB 3 — Drive Diagnostics
# -------------------------
with tab_diag:
    st.subheader("Google Drive Diagnostics")

    diag = {
        "has_GDRIVE_FOLDER_ID": bool(GDRIVE_FOLDER_ID),
        "has_SERVICE_ACCOUNT_JSON": bool(SERVICE_ACCOUNT_JSON),
        "sa_client_email": sa_email,
        "sa_project_id": sa_project,
        "sa_private_key_id_tail": (sa_key_id[-6:] if sa_key_id else None),
        "build_client_ok": gdrive_enabled and (gdrive_err is None),
        "build_client_error": gdrive_err,
        "folder_id": GDRIVE_FOLDER_ID,
    }

    # Step 1: about().get() — proves auth works
    about_ok = False
    about_err = None
    about_user = None
    if gdrive_enabled:
        try:
            about = drive.about().get(fields="user, kind").execute()
            about_ok = True
            about_user = about.get("user", {})
        except Exception as e:
            about_err = f"{type(e).__name__}: {e}"
    diag["about_ok"] = about_ok
    diag["about_error"] = about_err
    diag["about_user"] = about_user

    # Step 2: files().get on the folder ID — proves access to that folder
    folder_ok = False
    folder_err = None
    folder_meta = None
    if gdrive_enabled and GDRIVE_FOLDER_ID:
        try:
            folder_meta = drive.files().get(
                fileId=GDRIVE_FOLDER_ID,
                fields="id, name, mimeType, parents, driveId, capabilities, permissions",
                supportsAllDrives=True,
            ).execute()
            folder_ok = True
        except Exception as e:
            folder_err = f"{type(e).__name__}: {e}"
    diag["folder_get_ok"] = folder_ok
    diag["folder_get_error"] = folder_err
    diag["folder_meta"] = folder_meta

    # Step 3: list first 50 children (if folder ok)
    list_ok = False
    list_err = None
    children_sample = []
    if folder_ok:
        try:
            resp = drive.files().list(
                q=f"'{GDRIVE_FOLDER_ID}' in parents and trashed=false",
                fields="files(id, name, mimeType, driveId)",
                pageSize=50,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            children_sample = resp.get("files", [])
            list_ok = True
        except Exception as e:
            list_err = f"{type(e).__name__}: {e}"
    diag["list_children_ok"] = list_ok
    diag["list_children_error"] = list_err
    diag["children_sample"] = children_sample

    # Step 4: cache check on server
    cached_bn = sorted(str(p) for p in CACHE_ROOT.glob("bn*.csv"))
    cached_kn = sorted(str(p) for p in CACHE_ROOT.glob("kn*.csv"))
    diag["cache_dir"] = str(CACHE_ROOT)
    diag["cached_bn"] = cached_bn[:50]
    diag["cached_kn"] = cached_kn[:50]

    # Present summary + a copyable block
    st.markdown("**Summary:**")
    summary_lines = []
    summary_lines.append(f"- Secrets present: GDRIVE_FOLDER_ID={bool(GDRIVE_FOLDER_ID)}, SERVICE_ACCOUNT_JSON={bool(SERVICE_ACCOUNT_JSON)}")
    summary_lines.append(f"- Service account: {sa_email or 'None'} (project: {sa_project or 'None'})")
    if gdrive_err:
        summary_lines.append(f"- Build client: ❌ {gdrive_err}")
    else:
        summary_lines.append(f"- Build client: {'✅ OK' if gdrive_enabled else '❌ FAILED'}")
    summary_lines.append(f"- about().get(): {'✅ OK' if about_ok else '❌ ' + (about_err or '')}")
    if folder_meta:
        mt = folder_meta.get('mimeType')
        dn = folder_meta.get('name')
        did = folder_meta.get('driveId')
        summary_lines.append(f"- Folder access: ✅ name='{dn}', mimeType='{mt}', driveId='{did}'")
    else:
        summary_lines.append(f"- Folder access: {'✅' if folder_ok else '❌ ' + (folder_err or '')}")
    summary_lines.append(f"- Children list: {'✅ ' + str(len(children_sample)) + ' items' if list_ok else '❌ ' + (list_err or '')}")
    summary_lines.append(f"- Cached CSVs: bn={len(cached_bn)}, kn={len(cached_kn)} at {CACHE_ROOT}")

    st.text("\n".join(summary_lines))

    st.markdown("**Copy & paste this diagnostic for me:**")
    st.code(json.dumps(diag, indent=2), language="json")

    st.markdown("---")
    st.caption("Common fixes:")
    st.markdown(
        "- Ensure the **folder** (and its parent **Shared Drive**, if applicable) is shared with "
        f"**{sa_email or '[service account email]'}** (Viewer or higher).  \n"
        "- If the folder is in a **Shared Drive**, item-level sharing may be blocked—add the service account to the **Shared Drive** members.  \n"
        "- Double-check `GDRIVE_FOLDER_ID` is exactly the ID (the long string in the URL), not the URL itself.  \n"
        "- In Google Cloud Console → **APIs & Services**, ensure **Google Drive API** is **Enabled** for the project "
        f"**{sa_project or '[project]'}**."
    )
