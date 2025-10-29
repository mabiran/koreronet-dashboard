#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KōreroNET Dashboard (Google Drive-backed)

What this does:
- Uses Google Drive API (service account from st.secrets) to list bn*.csv / kn*.csv in your Folder ID.
- Downloads (and reuses) CSVs into a local cache (/tmp/koreronet_cache).
- Keeps your existing date indexing, heatmap, and verify-player logic unchanged as much as possible.
- Tab 1 = per-day heatmaps from BN/KN/Combined, filtered by confidence.
- Tab 2 = Verify snapshots under Backup/YYYMMDD_HHMMSS/{koreronet,birdnet} on Drive, with lazy WAV chunk download.
- Tab 3 = Quick Drive view + counts (handy to confirm folder contents).

Requirements: same as your diagnostics (google-api-python-client, google-auth, etc.)
"""

import os, io, glob, re, json
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------- UI setup ---------------
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
st.title("KōreroNET • Daily Dashboard")

# --------------- Local cache ---------------
CACHE_ROOT = Path("/tmp/koreronet_cache")
CSV_CACHE = CACHE_ROOT / "csv"
CHUNK_CACHE = CACHE_ROOT / "chunks"
CSV_CACHE.mkdir(parents=True, exist_ok=True)
CHUNK_CACHE.mkdir(parents=True, exist_ok=True)

# --------------- Google Drive client ---------------
def _normalize_private_key(pk: str) -> str:
    if not isinstance(pk, str):
        return pk
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    if "-----BEGIN PRIVATE KEY-----" in pk and "-----END PRIVATE KEY-----" in pk:
        if "-----BEGIN PRIVATE KEY-----\n" not in pk:
            pk = pk.replace("-----BEGIN PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----\n", 1)
        if "\n-----END PRIVATE KEY-----" not in pk:
            pk = pk.replace("-----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----", 1)
    return pk

def build_drive():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    sa_tbl = st.secrets.get("service_account")
    sa_json = st.secrets.get("SERVICE_ACCOUNT_JSON")
    if not sa_tbl and not sa_json:
        return None
    if sa_tbl:
        info = dict(sa_tbl)
    else:
        info = json.loads(sa_json)
    if "private_key" in info and isinstance(info["private_key"], str):
        info["private_key"] = _normalize_private_key(info["private_key"])
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID")
drive = build_drive() if GDRIVE_FOLDER_ID else None
DRIVE_ENABLED = bool(drive and GDRIVE_FOLDER_ID)

# --------------- Drive helpers ---------------
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    """List non-trashed children in a folder (supports Shared Drives)."""
    items, token = [], None
    while True:
        page_size = min(100, max_items - len(items))
        if page_size <= 0:
            break
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,md5Checksum,parents)",
            pageSize=page_size,
            pageToken=token,
            orderBy="folder,name_natural",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="user",  # works with parent filtering; for Shared Drive you can switch to "drive" with driveId
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return items

def find_child_folder_by_name(parent_id: str, name: str) -> Optional[Dict[str, Any]]:
    kids = list_children(parent_id, max_items=2000)
    for k in kids:
        if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name") == name:
            return k
    return None

def download_to(path: Path, file_id: str) -> Path:
    """Download Drive fileId to local path (overwrite if exists)."""
    from googleapiclient.http import MediaIoBaseDownload
    path.parent.mkdir(parents=True, exist_ok=True)
    req = drive.files().get_media(fileId=file_id)
    with open(path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return path

def ensure_csv_cached(file_meta: Dict[str, Any], subdir: str = "") -> Path:
    """Ensure a CSV is cached by name + md5 checksum."""
    name = file_meta["name"]
    md5 = file_meta.get("md5Checksum") or "nomd5"
    target_dir = CSV_CACHE / subdir if subdir else CSV_CACHE
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / name

    # If the same name exists but checksum changed, re-download (store alongside with same name).
    # For simplicity we re-download if file not present; checksum verification optional speed tradeoff.
    if not local_path.exists():
        download_to(local_path, file_meta["id"])
    return local_path

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
    """Ensure an audio chunk (e.g., .wav) with given name is cached from folder_id."""
    local_path = (CHUNK_CACHE / subdir / chunk_name)
    if local_path.exists():
        return local_path
    # find by exact name in the folder
    kids = list_children(folder_id, max_items=2000)
    for k in kids:
        if k.get("name") == chunk_name:
            try:
                download_to(local_path, k["id"])
                return local_path
            except Exception:
                return None
    return None

# --------------- Local fallback (unchanged paths) ---------------
DEFAULT_ROOT = r"G:\My Drive\From the node"
BACKUP_ROOT_DEFAULT = r"G:\My Drive\From the node\Backup"
ROOT_LOCAL = os.environ.get("KORERONET_DATA_ROOT", DEFAULT_ROOT)
BACKUP_ROOT_LOCAL = os.environ.get("KORERONET_BACKUP_ROOT", BACKUP_ROOT_DEFAULT)

# --------------- CSV discovery (Drive or Local) ---------------
@st.cache_data(show_spinner=False)
def list_csvs_drive(folder_id: str) -> Tuple[List[Path], List[Path]]:
    kids = list_children(folder_id, max_items=2000)
    bn = [k for k in kids if k.get("name", "").lower().startswith("bn") and k.get("name", "").lower().endswith(".csv")]
    kn = [k for k in kids if k.get("name", "").lower().startswith("kn") and k.get("name", "").lower().endswith(".csv")]
    bn_paths = [ensure_csv_cached(m) for m in bn]
    kn_paths = [ensure_csv_cached(m) for m in kn]
    # sort by name to keep behavior stable
    return (sorted(bn_paths, key=lambda p: p.name), sorted(kn_paths, key=lambda p: p.name))

@st.cache_data(show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    return bn_paths, kn_paths

def list_csvs_any() -> Tuple[List[Path], List[Path], str]:
    if DRIVE_ENABLED:
        bn, kn = list_csvs_drive(GDRIVE_FOLDER_ID)
        return bn, kn, "drive"
    else:
        bn, kn = list_csvs_local(ROOT_LOCAL)
        # cast to Path for uniformity
        return [Path(p) for p in bn], [Path(p) for p in kn], "local"

# --------------- Index ALL dates present inside each CSV (robust to DD/MM/YYYY) ---------------
@st.cache_data(show_spinner=False)
def extract_dates_from_csv(path: str | Path) -> List[date]:
    path = str(path)
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
def build_date_index(bn_paths: List[Path], kn_paths: List[Path]) -> Tuple[Dict[date, List[str]], Dict[date, List[str]]]:
    bn_idx, kn_idx = {}, {}
    for p in bn_paths:
        for d in extract_dates_from_csv(p):
            bn_idx.setdefault(d, []).append(str(p))
    for p in kn_paths:
        for d in extract_dates_from_csv(p):
            kn_idx.setdefault(d, []).append(str(p))
    return bn_idx, kn_idx

# --------------- Normalisation + plotting (unchanged) ---------------
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
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(str(path))

# --------------- Verify tab (Drive snapshots OR local) ---------------
SNAP_RE = re.compile(r"^(\d{8})_(\d{6})$")

def _parse_snapshot(name: str) -> Optional[datetime]:
    m = SNAP_RE.match(name)
    if not m: return None
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")

@st.cache_data(show_spinner=False)
def list_snapshots_local(backup_root: str) -> List[Tuple[datetime, str, str]]:
    if not os.path.isdir(backup_root): return []
    items = []
    for n in os.listdir(backup_root):
        p = os.path.join(backup_root, n)
        if os.path.isdir(p):
            dt = _parse_snapshot(n)
            if dt: items.append((dt, n, p))
    items.sort(key=lambda x: x[0], reverse=True)
    return items

@st.cache_data(show_spinner=False)
def list_snapshots_drive(root_folder_id: str) -> List[Tuple[datetime, str, str]]:
    """Find 'Backup' folder under root, then list snapshot subfolders."""
    backup = find_child_folder_by_name(root_folder_id, "Backup")
    if not backup:
        return []
    snap_folders = [k for k in list_children(backup["id"], max_items=2000)
                    if k.get("mimeType") == "application/vnd.google-apps.folder" and SNAP_RE.match(k["name"] or "")]
    out = []
    for k in snap_folders:
        dt = _parse_snapshot(k["name"])
        if dt:
            out.append((dt, k["name"], k["id"]))
    out.sort(key=lambda x: x[0], reverse=True)
    return out  # [(dt, folder_name, folder_id)]

def list_snapshots_any() -> Tuple[List[Tuple[datetime, str, str]], str]:
    if DRIVE_ENABLED:
        return list_snapshots_drive(GDRIVE_FOLDER_ID), "drive"
    else:
        snaps = list_snapshots_local(BACKUP_ROOT_LOCAL)
        # for local we keep folder path string as third element
        return snaps, "local"

def _gather_csvs_local(snapshot_path: str, src_mode: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
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

def _gather_csvs_drive(snapshot_folder_id: str, src_mode: str) -> Tuple[List[Tuple[str, Path, str]], Optional[str]]:
    """Return list of tuples: (kind, local_csv_path, chunk_drive_folder_id)"""
    kids = list_children(snapshot_folder_id, max_items=2000)
    sub_kn = [k for k in kids if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name") == "koreronet"]
    sub_bn = [k for k in kids if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name") == "birdnet"]

    todo: List[Tuple[str, Path, str]] = []
    if src_mode in ("KōreroNET (KN)", "Combined") and sub_kn:
        kn_id = sub_kn[0]["id"]
        kn_kids = list_children(kn_id, max_items=2000)
        for f in kn_kids:
            if f.get("name", "").lower().endswith(".csv"):
                lp = ensure_csv_cached(f, subdir=f"snap_{snapshot_folder_id}/koreronet")
                todo.append(("kn", lp, kn_id))
    if src_mode in ("BirdNET (BN)", "Combined") and sub_bn:
        bn_id = sub_bn[0]["id"]
        bn_kids = list_children(bn_id, max_items=2000)
        for f in bn_kids:
            if f.get("name", "").lower().endswith(".csv"):
                lp = ensure_csv_cached(f, subdir=f"snap_{snapshot_folder_id}/birdnet")
                todo.append(("bn", lp, bn_id))

    if src_mode == "Combined":
        has_kn = any(k == "kn" for k, _, _ in todo)
        has_bn = any(k == "bn" for k, _, _ in todo)
        if not (has_kn and has_bn):
            return [], "Combined requires both 'koreronet' and 'birdnet' folders."
    return sorted(todo, key=lambda t: str(t[1]).lower()), None

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
        return pd.DataFrame(columns=["Label","Confidence","ActualTime","Kind","ChunkPath","Start_s","End_s","ChunkName","ChunkDriveFolderId"])
    use_off = df["det_start"].where(df["det_start"].notna(), df["start_s"])
    times = [_actual_time(s, o) for s, o in zip(df["src"], use_off)]
    out = pd.DataFrame({
        "Label": df["label"].astype(str),
        "Confidence": df["prob"].astype(float),
        "ActualTime": times,
        "Kind": df["Kind"],
        "ChunkPath": df.get("ChunkPath", pd.Series([""]*len(df))),
        "ChunkName": df.get("ChunkName", pd.Series([""]*len(df))),
        "ChunkDriveFolderId": df.get("ChunkDriveFolderId", pd.Series([""]*len(df))),
        "Start_s": df["start_s"],
        "End_s": df["end_s"],
    })
    return out.dropna(subset=["ActualTime"]).sort_values("ActualTime").reset_index(drop=True)

# =========================================
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "📁 Drive"])

# -------------------------
# TAB 1 — calendar + 0.90 default + strict per-day filter (Drive or Local)
# -------------------------
with tab1:
    bn_paths, kn_paths, mode = list_csvs_any()
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
        if st.form_submit_button("Update"):
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
# TAB 2 — Verify (Drive snapshots or local), 0.90 default, calendar filter inside loaded detections
# -------------------------
with tab_verify:
    st.subheader("Verify recordings")

    snaps, s_mode = list_snapshots_any()
    if not snaps:
        st.warning("No snapshot folders like YYYYMMDD_HHMMSS found (Backup folder).")
        st.stop()

    snap_name = st.selectbox("Snapshot", options=[n for _, n, _ in snaps], index=0, key="verify_snap")
    snap_third = next(t for _, n, t in snaps if n == snap_name)  # drive folder ID or local path

    src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0, key="verify_src")
    min_conf_v = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01, key="verify_conf")

    ctx = {"snap": snap_name, "src": src_mode_v, "conf": float(min_conf_v)}
    if st.session_state.get("v_ctx") != ctx:
        st.session_state["v_ctx"] = ctx
        st.session_state["v_loaded"] = False
        st.session_state["v_df"] = pd.DataFrame()

    def _load_with_progress_local(csv_list: List[Tuple[str, str]]) -> pd.DataFrame:
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
                out["ChunkName"] = out["chunk_file"].apply(lambda f: os.path.basename(f) if isinstance(f, str) else "")
                out["ChunkDriveFolderId"] = ""  # local mode
                frames.append(out)
            except Exception:
                pass
            prog.progress(i/n)
        status.text("Processing…")
        prog.progress(1.0)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_with_progress_drive(csv_list: List[Tuple[str, Path, str]]) -> pd.DataFrame:
        """Each item: (kind, local_csv_path, chunk_drive_folder_id)"""
        prog = st.progress(0.0)
        status = st.empty()
        frames, n = [], max(1, len(csv_list))
        for i, (kind, path, chunk_folder_id) in enumerate(csv_list, 1):
            status.text(f"Loading {i}/{n}: {os.path.basename(str(path))}")
            try:
                df = pd.read_csv(str(path), sep=None, engine="python")
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
                # For Drive mode, we set ChunkPath lazily; keep name and folder id:
                out["ChunkName"] = out["chunk_file"].apply(lambda f: os.path.basename(f) if isinstance(f, str) else "")
                out["ChunkDriveFolderId"] = chunk_folder_id
                out["ChunkPath"] = ""  # will be resolved at play-time
                frames.append(out)
            except Exception:
                pass
            prog.progress(i/n)
        status.text("Processing…")
        prog.progress(1.0)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # LOAD button
    if st.button("Load detections", type="primary"):
        if s_mode == "local":
            csv_list, err = _gather_csvs_local(snap_third, src_mode_v)
        else:
            csv_list, err = _gather_csvs_drive(snap_third, src_mode_v)
        if err:
            st.warning(err)
        elif not csv_list:
            st.warning("No CSV files found in the selected snapshot/source.")
        else:
            if s_mode == "local":
                raw = _load_with_progress_local(csv_list)
            else:
                raw = _load_with_progress_drive(csv_list)
            std_v = _std_verify(raw)
            std_v = std_v[std_v["Confidence"] >= float(min_conf_v)]
            st.session_state["v_df"] = std_v
            st.session_state["v_loaded"] = True
            # reset playlist indices
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("v_idx::"):
                    del st.session_state[k]

    if not st.session_state.get("v_loaded", False):
        st.info("Adjust settings and click **Load detections** to enable the player.")
        st.stop()

    std_v = st.session_state.get("v_df", pd.DataFrame())
    if std_v.empty:
        st.warning("No detections after filtering."); st.stop()

    # Calendar filter
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

    # Build playlist
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

    # Audio player (supports both local and Drive-lazy modes)
    def _play_audio(row: pd.Series, auto: bool):
        # Prefer existing ChunkPath if valid
        path_str = str(row.get("ChunkPath", "") or "")
        if path_str and os.path.isfile(path_str):
            try:
                with open(path_str, "rb") as f:
                    data = f.read()
            except Exception:
                data = None
        else:
            data = None

        # If not present and we have Drive info, fetch lazily
        if (data is None) and DRIVE_ENABLED:
            chunk_name = str(row.get("ChunkName", "") or "")
            folder_id = str(row.get("ChunkDriveFolderId", "") or "")
            if chunk_name and folder_id:
                # choose a subdir by snapshot & source for organization
                subdir = f"{snap_name}/{row.get('Kind','UNK')}"
                cached = ensure_chunk_cached(chunk_name, folder_id, subdir=subdir)
                if cached and cached.exists():
                    try:
                        with open(cached, "rb") as f:
                            data = f.read()
                    except Exception:
                        data = None

        if data is None:
            st.warning("Cannot open audio chunk (not found or download failed).")
            return

        if auto:
            import base64
            b64 = base64.b64encode(data).decode()
            st.markdown(f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>', unsafe_allow_html=True)
        else:
            st.audio(data, format="audio/wav")

    _play_audio(row, autoplay)

# -------------------------
# TAB 3 — Drive quick view (optional)
# -------------------------
with tab3:
    if not DRIVE_ENABLED:
        st.info("Drive not configured. Set GDRIVE_FOLDER_ID and service_account in Streamlit secrets.")
    else:
        st.caption("Top-level of your Google Drive Folder ID")
        max_items = st.slider("Max items to show", 50, 1000, 300, 50)
        with st.spinner("Listing…"):
            kids = list_children(GDRIVE_FOLDER_ID, max_items=max_items)
        if not kids:
            st.warning("No items found under this Folder ID.")
        else:
            # quick counts
            bn = [k for k in kids if k.get("name","").lower().startswith("bn") and k.get("name","").lower().endswith(".csv")]
            kn = [k for k in kids if k.get("name","").lower().startswith("kn") and k.get("name","").lower().endswith(".csv")]
            st.info(f"bn*.csv: {len(bn)} | kn*.csv: {len(kn)}")
            st.dataframe(
                [{"name": k.get("name"), "id": k.get("id"), "mimeType": k.get("mimeType"),
                  "modifiedTime": k.get("modifiedTime"), "size": k.get("size")} for k in kids],
                use_container_width=True, hide_index=True
            )
