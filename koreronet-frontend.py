#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KōreroNET Dashboard (with splash + Drive)
# ------------------------------------------------------------
# - Splash screen (“KōreroNET” + “AUT”) for ~2s, then main UI
# - Node Select at top (single option: Auckland-Orākei)
# - Tab 1: Root CSV heatmaps (Drive or local) with calendar + min confidence
# - Tab 2: Verify using snapshot date (Backup/YYYYMMDD_HHMMSS) and master CSVs;
#          on-demand audio chunk fetch from Drive (no full directory downloads).
# - Tab 3: Power graph from "Power logs" (Drive), stitches latest N logs;
#          dual y-axes: SoC_i (%) and Wh.
#
# Streamlit secrets required:
#   GDRIVE_FOLDER_ID = "your_root_folder_id"
#   [service_account]
#   type = "service_account"
#   project_id = "..."
#   private_key_id = "..."
#   private_key = (paste PEM with real newlines, no \n escapes)
#   client_email = "...@...iam.gserviceaccount.com"
#   client_id = "..."
#   auth_uri = "https://accounts.google.com/o/oauth2/auth"
#   token_uri = "https://oauth2.googleapis.com/token"
#   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
#   client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/...."
#   universe_domain = "googleapis.com"
# ------------------------------------------------------------

import os, io, re, glob, json, time
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# New pipeline start date: detection files produced on and after this date
# follow the new unified master CSV format.  Use this constant to decide
# whether to load old bn/kn CSVs or the new *_birdnet_master.csv and
# *_koreronet_master.csv files.  See process.py for details.
NEW_PIPELINE_START = date(2025, 10, 31)

# ─────────────────────────────────────────────────────────────
# Page style
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="KōreroNET Dashboard", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:1rem;}
.center-wrap {display:flex; align-items:center; justify-content:center; min-height:65vh; text-align:center;}
.brand-title {font-size: clamp(48px, 8vw, 96px); font-weight: 800; letter-spacing: .02em;}
.brand-sub {font-size: clamp(28px, 4vw, 48px); font-weight: 600; opacity:.9; margin-top:.4rem;}
.fade-enter {animation: fadeIn 400ms ease forwards;}
.fade-exit  {animation: fadeOut 400ms ease forwards;}
@keyframes fadeIn { from {opacity:0} to {opacity:1} }
@keyframes fadeOut { from {opacity:1} to {opacity:0} }
.pulse {position:relative; width:14px; height:14px; margin:18px auto 0; border-radius:50%; background:#16a34a; box-shadow:0 0 0 rgba(22,163,74,.7); animation: pulse 1.6s infinite;}
@keyframes pulse { 0%{ box-shadow:0 0 0 0 rgba(22,163,74,.7);} 70%{ box-shadow:0 0 0 22px rgba(22,163,74,0);} 100%{ box-shadow:0 0 0 0 rgba(22,163,74,0);} }
.stTabs [role="tablist"] {gap:.5rem;}
.stTabs [role="tab"] {padding:.6rem 1rem; border-radius:999px; border:1px solid #3a3a3a;}
.small {font-size:0.9rem; opacity:0.85;}
</style>
""", unsafe_allow_html=True)
# ---- Chunk filename parsing / cache helpers ----
CHUNK_RE = re.compile(
    r"^(?P<root>\d{8}_\d{6})__(?P<tag>bn|kn)_(?P<s>\d+\.\d{2})_(?P<e>\d+\.\d{2})__(?P<label>.+?)__p(?P<conf>\d+\.\d{2})\.wav$",
    re.IGNORECASE,
)

def _parse_chunk_filename(name: str) -> Optional[Dict[str, Any]]:
    m = CHUNK_RE.match(name or "")
    if not m:
        return None
    try:
        return {
            "root":  m.group("root"),
            "tag":   m.group("tag").lower(),
            "s":     float(m.group("s")),
            "e":     float(m.group("e")),
            "label": m.group("label"),
            "conf":  float(m.group("conf")),
        }
    except Exception:
        return None

def _folder_children_cached(folder_id: str) -> List[Dict[str, Any]]:
    key = f"drive_kids::{folder_id}"
    if key not in st.session_state:
        st.session_state[key] = list_children(folder_id, max_items=2000)
    return st.session_state[key]

# ─────────────────────────────────────────────────────────────
# Simple splash gate
# ─────────────────────────────────────────────────────────────
if "splash_done" not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            """
            <div class="center-wrap fade-enter">
              <div>
                <div class="brand-title">KōreroNET</div>
                <div class="brand-sub">AUT</div>
                <div class="pulse"></div>
                <div class="small" style="margin-top:10px;">initialising…</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    time.sleep(2.0)
    st.session_state["splash_done"] = True
    st.rerun()

# ─────────────────────────────────────────────────────────────
# Caches & local fallback
# ─────────────────────────────────────────────────────────────
CACHE_ROOT   = Path("/tmp/koreronet_cache")
CSV_CACHE    = CACHE_ROOT / "csv"
CHUNK_CACHE  = CACHE_ROOT / "chunks"
POWER_CACHE  = CACHE_ROOT / "power"
for _p in (CSV_CACHE, CHUNK_CACHE, POWER_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

DEFAULT_ROOT = r"G:\My Drive\From the node"
ROOT_LOCAL   = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)

# Node Select (top bar)
node = st.selectbox("Node Select", ["Auckland-Orākei"], index=0, key="node_select_top")

# ─────────────────────────────────────────────────────────────
# Secrets / Drive builders
# ─────────────────────────────────────────────────────────────
GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID", None)

def _normalize_private_key(pk: str) -> str:
    if not isinstance(pk, str): return pk
    if "\\n" in pk: pk = pk.replace("\\n", "\n")
    if "-----BEGIN PRIVATE KEY-----" in pk and "-----END PRIVATE KEY-----" in pk:
        if "-----BEGIN PRIVATE KEY-----\n" not in pk:
            pk = pk.replace("-----BEGIN PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----\n", 1)
        if "\n-----END PRIVATE KEY-----" not in pk:
            pk = pk.replace("-----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----", 1)
    return pk

def _build_drive_client():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        sa_tbl  = st.secrets.get("service_account")
        sa_json = st.secrets.get("SERVICE_ACCOUNT_JSON")
        if not sa_tbl and not sa_json:
            return None
        info = dict(sa_tbl) if sa_tbl else json.loads(sa_json)
        if "private_key" in info:
            info["private_key"] = _normalize_private_key(info["private_key"])
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None

def get_drive_client():
    if "drive_client" in st.session_state:
        return st.session_state["drive_client"]
    client = _build_drive_client() if GDRIVE_FOLDER_ID else None
    st.session_state["drive_client"] = client
    return client

def drive_enabled() -> bool:
    return bool(GDRIVE_FOLDER_ID and get_drive_client())

# ─────────────────────────────────────────────────────────────
# Drive helpers
# ─────────────────────────────────────────────────────────────
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    drive = get_drive_client()
    if not drive: return []
    items, token = [], None
    while True:
        page_size = min(100, max_items - len(items))
        if page_size <= 0: break
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields=("nextPageToken, files(id,name,mimeType,modifiedTime,size,md5Checksum,parents)"),
            pageSize=page_size,
            pageToken=token,
            orderBy="folder,name_natural",
            includeItemsFromAllDrives=True, supportsAllDrives=True, corpora="allDrives",
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token: break
    return items

def download_to(path: Path, file_id: str) -> Path:
    drive = get_drive_client()
    from googleapiclient.http import MediaIoBaseDownload
    path.parent.mkdir(parents=True, exist_ok=True)
    req = drive.files().get_media(fileId=file_id)
    with open(path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return path

# ----------------------------------------------------------------------------
# Master CSV helpers (new pipeline)
# ----------------------------------------------------------------------------
#
# In the new pipeline (from 31 Oct 2025 onward), detections are written to
# master CSVs in the "From the node" folder.  These files are named with a
# timestamp prefix followed by either `_birdnet_master.csv` or
# `_koreronet_master.csv`.  Each row contains the audio clip name, the
# actual (wall-clock) start time, the detected label and a probability.  The
# helper functions below locate these master files in Drive or a local root,
# read them into unified DataFrames, and build indices for calendar
# selection.

def list_master_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    """
    Return local master CSV paths from the given root directory.  Files
    ending with `_birdnet_master.csv` are considered BirdNET masters and
    those ending with `_koreronet_master.csv` are considered KōreroNET
    masters.
    """
    bn_paths = sorted(glob.glob(os.path.join(root, "*_birdnet_master.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "*_koreronet_master.csv")))
    return bn_paths, kn_paths

def list_master_csvs_drive_root(folder_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return Drive file metadata for master CSVs under the given folder.  A
    master CSV is identified by its suffix: `_birdnet_master.csv` or
    `_koreronet_master.csv`.  The returned lists are sorted by filename.
    """
    kids = list_children(folder_id, max_items=2000)
    bn = [k for k in kids if k.get("name", "").lower().endswith("_birdnet_master.csv")]
    kn = [k for k in kids if k.get("name", "").lower().endswith("_koreronet_master.csv")]
    bn.sort(key=lambda m: m.get("name", ""))
    kn.sort(key=lambda m: m.get("name", ""))
    return bn, kn

def _read_master_standard(path: Path, kind: Optional[str] = None) -> pd.DataFrame:
    """
    Read a master CSV (new pipeline) and return a DataFrame with unified
    column names.  Attempts to coerce varying header names into the
    standard set: Clip, ActualStartTime, Label, Probability.  A 'Kind'
    column is added if `kind` is provided ("BN" or "KN").  The caller
    should provide `kind` when the file identity is known (e.g. via
    list_master_csvs_* functions).
    """
    try:
        df = pd.read_csv(str(path))
    except Exception:
        return pd.DataFrame(columns=["Clip", "ActualStartTime", "Label", "Probability"])
    # map possible column names to standard names
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands: str) -> Optional[str]:
        for cand in cands:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None
    clip_col = pick("clip", "chunk", "file", "wav", "chunkname")
    time_col = pick("actualstarttime", "actual time", "actualtime", "actual_time", "time", "timestamp")
    label_col = pick("label", "common name", "common_name", "species", "class")
    prob_col = pick("probability", "prob", "confidence", "p")
    if not clip_col or not time_col or not label_col:
        return pd.DataFrame(columns=["Clip", "ActualStartTime", "Label", "Probability"])
    out = pd.DataFrame({
        "Clip": df[clip_col].astype(str),
        "ActualStartTime": pd.to_datetime(df[time_col], errors="coerce", dayfirst=True),
        "Label": df[label_col].astype(str).str.strip(),
        # Probability may not exist (older versions may have Confidence); default to NaN
        "Probability": pd.to_numeric(df.get(prob_col, df.get(pick("confidence", "probability"), np.nan)), errors="coerce")
    })
    # Remove rows with missing times or labels
    out = out.dropna(subset=["ActualStartTime", "Label"])
    if kind is not None:
        out["Kind"] = kind
    return out

def build_date_index_from_masters(paths: List[Path]) -> Dict[date, List[Path]]:
    """
    For a list of master CSV paths, build an index mapping each date to
    the list of files that contain detections on that date.  This helper
    loads only the ActualStartTime column to minimize I/O.
    """
    idx: Dict[date, List[Path]] = {}
    for p in paths:
        try:
            for chunk in pd.read_csv(p, usecols=[0, 1], chunksize=5000):
                # column names are unknown; pick by index: Clip and ActualStartTime
                # attempt to parse second column as datetime
                s = pd.to_datetime(chunk.iloc[:, 1], errors="coerce", dayfirst=True)
                for ts in s.dropna():
                    idx.setdefault(ts.date(), []).append(p)
        except Exception:
            try:
                df = _read_master_standard(p)
                for d in df["ActualStartTime"].dt.date.unique():
                    idx.setdefault(d, []).append(p)
            except Exception:
                pass
    # deduplicate file lists
    for key in list(idx.keys()):
        idx[key] = sorted(set(idx[key]), key=lambda x: x.name)
    return idx

def load_master_for_day(paths: List[Path], day_selected: date) -> pd.DataFrame:
    """
    Given a list of master CSV paths and a selected date, load all
    detections occurring on that date.  Returns a DataFrame with
    Clip, ActualStartTime, Label, Probability and Kind columns if present.
    """
    frames = []
    for p in paths:
        try:
            kind = None
            nm = p.name.lower()
            if nm.endswith("_birdnet_master.csv"):
                kind = "BN"
            elif nm.endswith("_koreronet_master.csv"):
                kind = "KN"
            df = _read_master_standard(p, kind=kind)
            if df.empty:
                continue
            df_day = df[df["ActualStartTime"].dt.date == day_selected]
            if not df_day.empty:
                frames.append(df_day)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Clip", "ActualStartTime", "Label", "Probability", "Kind"])

def find_backup_audio_folders(root_id: str) -> Dict[str, str]:
    """
    Locate the backup audio folders for BirdNET and KōreroNET under the
    Drive root.  Returns a mapping of kind -> folder id.  If the
    expected folders are not found, the root_id is used as a fallback.
    """
    # find 'Backup' folder
    kids = list_children(root_id, max_items=2000)
    backup = next((k for k in kids if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name", "").lower() == "backup"), None)
    if not backup:
        return {"BN": root_id, "KN": root_id}
    subkids = list_children(backup["id"], max_items=2000)
    id_bn = next((k["id"] for k in subkids if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name", "").lower() == "birdnet"), backup["id"])
    id_kn = next((k["id"] for k in subkids if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name", "").lower() == "koreronet"), backup["id"])
    return {"BN": id_bn, "KN": id_kn}

def ensure_csv_cached(meta: Dict[str, Any], subdir: str) -> Path:
    local_path = (CSV_CACHE / subdir / meta["name"])
    if not local_path.exists():
        download_to(local_path, meta["id"])
    return local_path

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
    """
    Try exact name first. If missing, fuzzy-match by (root, tag, label) and choose
    the chunk whose [s,e] window contains the requested [s_t,e_t] (±0.75s tolerance).
    """
    local_path = CHUNK_CACHE / subdir / chunk_name
    if local_path.exists():
        return local_path

    # 1) Exact lookup
    kids = _folder_children_cached(folder_id)
    name_to_id = {k.get("name"): k.get("id") for k in kids}
    if chunk_name in name_to_id:
        try:
            download_to(local_path, name_to_id[chunk_name])
            return local_path
        except Exception:
            return None

    # 2) Fuzzy fallback
    # Parse the target request so we know what we’re looking for
    target = _parse_chunk_filename(chunk_name)
    if not target:
        return None

    root_t  = target["root"]
    tag_t   = target["tag"]
    s_t     = target["s"]
    e_t     = target["e"]
    label_t = target["label"].lower()

    # Gather candidate WAVs in the folder
    candidates = []
    for k in kids:
        nm = k.get("name","")
        if not nm.lower().endswith(".wav"):
            continue
        info = _parse_chunk_filename(nm)
        if not info:
            continue
        # same recording root, same tag (bn/kn), same label (case-insensitive, underscores same)
        if info["root"] == root_t and info["tag"] == tag_t and info["label"].lower() == label_t:
            candidates.append((info, k.get("id"), nm))

    if not candidates:
        return None

    # Choose the candidate whose [s,e] contains [s_t,e_t] with a small tolerance
    tol = 0.75  # seconds
    def score(cinfo):
        s_c, e_c = cinfo["s"], cinfo["e"]
        # containment check
        contains = (s_c - tol) <= s_t and (e_c + tol) >= e_t
        # closeness (center distance) as tie-breaker
        cen_diff = abs(((s_c + e_c) * 0.5) - ((s_t + e_t) * 0.5))
        return (0 if contains else 1, cen_diff)

    candidates.sort(key=lambda it: score(it[0]))
    best_info, best_id, best_name = candidates[0]

    try:
        download_to(local_path, best_id)
        # Optional: let the user know we had to fuzzy-match
        st.caption(f"⚠️ Used fuzzy match: requested `{chunk_name}` → found `{best_name}`")
        return local_path
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Tab 1 (root CSV heatmap)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    return bn_paths, kn_paths

@st.cache_data(show_spinner=False)
def list_csvs_drive_root(folder_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kids = list_children(folder_id, max_items=2000)
    bn = [k for k in kids if k.get("name","").lower().startswith("bn") and k.get("name","").lower().endswith(".csv")]
    kn = [k for k in kids if k.get("name","").lower().startswith("kn") and k.get("name","").lower().endswith(".csv")]
    bn.sort(key=lambda m: m.get("name","")); kn.sort(key=lambda m: m.get("name",""))
    return bn, kn

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
                if "ActualTime" not in chunk.columns: continue
                s = pd.to_datetime(chunk["ActualTime"], errors="coerce", dayfirst=True)
                dates.update(ts.date() for ts in s.dropna())
        except Exception:
            return []
    return sorted(dates)

@st.cache_data(show_spinner=False)
def build_date_index(paths: List[Path]) -> Dict[date, List[str]]:
    idx: Dict[date, List[str]] = {}
    for p in paths:
        for d in extract_dates_from_csv(p):
            idx.setdefault(d, []).append(str(p))
    return idx

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
    hour_labels = {h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)}
    order = [hour_labels[h] for h in range(24)]
    df_f["HourLabel"] = df_f["Hour"].map(hour_labels)
    pivot = df_f.groupby(["Label","HourLabel"]).size().unstack(fill_value=0).astype(int)
    for lbl in order:
        if lbl not in pivot.columns: pivot[lbl] = 0
    pivot = pivot[order]
    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values(ascending=False).index]
    fig = px.imshow(
        pivot.values, x=pivot.columns, y=pivot.index,
        color_continuous_scale="RdYlBu_r",
        labels=dict(x="Hour (AM/PM)", y="Species (common name / label)", color="Detections"),
        text_auto=True, aspect="auto", title=title,
    )
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(str(path))

def calendar_pick(available_days: List[date], label: str, help_txt: str = "") -> date:
    if not available_days:
        st.stop()
    available_days = sorted(available_days)
    d_min, d_max = available_days[0], available_days[-1]
    d_val = st.date_input(label, value=d_max, min_value=d_min, max_value=d_max, help=help_txt)
    if d_val not in set(available_days):
        earlier = [x for x in available_days if x <= d_val]
        if earlier:
            d_val = earlier[-1]
            st.info(f"No data on chosen date; showing {d_val.isoformat()} (nearest earlier).")
        else:
            later = [x for x in available_days if x >= d_val]
            d_val = later[0]
            st.info(f"No data on chosen date; showing {d_val.isoformat()} (nearest later).")
    return d_val

# ─────────────────────────────────────────────────────────────
# Snapshot/master logic for Tab 2
# ─────────────────────────────────────────────────────────────
SNAP_RE = re.compile(r"^(\d{8})_(\d{6})$", re.IGNORECASE)

def _sanitize_label(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s

def _compose_chunk_name(kind: str, wav_base: str, start: float, end: float, label: str, conf: float) -> str:
    root = Path(wav_base).stem
    start_s = f"{float(start):06.2f}"
    end_s   = f"{float(end):06.2f}"
    lab     = _sanitize_label(label)
    pc      = f"{float(conf):.2f}"
    tag     = "bn" if kind.lower()=="bn" else "kn"
    return f"{root}__{tag}_{start_s}_{end_s}__{lab}__p{pc}.wav"

def _parse_date_from_snapname(name: str) -> Optional[date]:
    m = SNAP_RE.match(name or "")
    if not m: return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except Exception:
        return None

def _find_backup_folder(root_folder_id: str) -> Optional[Dict[str, Any]]:
    kids = list_children(root_folder_id, max_items=2000)
    for k in kids:
        if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name","").lower() == "backup":
            return k
    return None

def _find_chunk_dirs(snapshot_id: str) -> Dict[str, str]:
    kids = list_children(snapshot_id, max_items=2000)
    kn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "koreronet" in k.get("name","").lower()]
    bn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "birdnet"   in k.get("name","").lower()]
    return {"KN": (kn[0]["id"] if kn else snapshot_id), "BN": (bn[0]["id"] if bn else snapshot_id)}

def _match_master_name(name: str, kind: str) -> bool:
    n = name.lower()
    if kind == "KN":
        return (("koreronet" in n) and ("detect" in n) and n.endswith(".csv"))
    else:
        return (("birdnet" in n) and ("detect" in n) and n.endswith(".csv"))

def _find_master_anywhere(snapshot_id: str, kind: str) -> Optional[Dict[str, Any]]:
    root_kids = list_children(snapshot_id, max_items=2000)
    files_only = [f for f in root_kids if f.get("mimeType") != "application/vnd.google-apps.folder"]
    cands = [f for f in files_only if _match_master_name(f.get("name",""), kind)]
    if cands:
        cands.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
        return dict(cands[0])
    subfolders = [f for f in root_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]
    for sf in subfolders:
        sub_files = list_children(sf["id"], max_items=2000)
        sub_files = [f for f in sub_files if f.get("mimeType") != "application/vnd.google-apps.folder"]
        c2 = [f for f in sub_files if _match_master_name(f.get("name",""), kind)]
        if c2:
            c2.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
            return dict(c2[0])
    return None

@st.cache_data(show_spinner=True)
def build_master_index_by_snapshot_date(root_folder_id: str) -> pd.DataFrame:
    backup = _find_backup_folder(root_folder_id)
    if not backup:
        return pd.DataFrame(columns=[
            "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
            "ChunkDriveFolderId","SnapId","SnapName"
        ])
    snaps = [k for k in list_children(backup["id"], max_items=2000)
             if k.get("mimeType")=="application/vnd.google-apps.folder" and SNAP_RE.match(k.get("name",""))]
    snaps.sort(key=lambda m: m.get("name",""), reverse=True)

    rows: List[Dict[str,Any]] = []
    for sn in snaps:
        snap_id, snap_name = sn["id"], sn["name"]
        snap_date = _parse_date_from_snapname(snap_name)
        if not snap_date: continue
        chunk_dirs = _find_chunk_dirs(snap_id)

        # KN master
        kn_meta = _find_master_anywhere(snap_id, "KN")
        if kn_meta:
            kn_csv = ensure_csv_cached(kn_meta, subdir=f"snap_{snap_id}/koreronet")
            try:
                df = pd.read_csv(kn_csv)
                for _, r in df.iterrows():
                    wav = os.path.basename(str(r.get("File","")))
                    start = float(r.get("Start", r.get("Start (s)", np.nan)))
                    end   = float(r.get("End",   r.get("End (s)",   np.nan)))
                    lab   = str(r.get("Label","Unknown"))
                    conf  = float(r.get("Confidence", np.nan))
                    rows.append({
                        "Date": snap_date, "Kind": "KN", "Label": lab, "Confidence": conf,
                        "Start": start, "End": end, "WavBase": wav,
                        "ChunkName": _compose_chunk_name("kn", wav, start, end, lab, conf),
                        "ChunkDriveFolderId": chunk_dirs["KN"],
                        "SnapId": snap_id, "SnapName": snap_name,
                    })
            except Exception:
                pass

        # BN master
        bn_meta = _find_master_anywhere(snap_id, "BN")
        if bn_meta:
            bn_csv = ensure_csv_cached(bn_meta, subdir=f"snap_{snap_id}/birdnet")
            try:
                df = pd.read_csv(bn_csv)
                for _, r in df.iterrows():
                    wav = os.path.basename(str(r.get("File","")))
                    start = float(r.get("Start (s)", r.get("Start", np.nan)))
                    end   = float(r.get("End (s)",   r.get("End",   np.nan)))
                    lab   = str(r.get("Common name", r.get("Label","Unknown")))
                    conf  = float(r.get("Confidence", np.nan))
                    rows.append({
                        "Date": snap_date, "Kind": "BN", "Label": lab, "Confidence": conf,
                        "Start": start, "End": end, "WavBase": wav,
                        "ChunkName": _compose_chunk_name("bn", wav, start, end, lab, conf),
                        "ChunkDriveFolderId": chunk_dirs["BN"],
                        "SnapId": snap_id, "SnapName": snap_name,
                    })
            except Exception:
                pass

    if not rows:
        return pd.DataFrame(columns=[
            "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
            "ChunkDriveFolderId","SnapId","SnapName"
        ])
    out = pd.DataFrame(rows)
    out.sort_values(["Date","Kind","Label"], ascending=[False, True, True], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "⚡ Power"])

# =========================
# TAB 1 — Detections (root)
# =========================
with tab1:
    # Display progress while scanning for CSVs
    center = st.empty()
    with center.container():
        st.markdown('<div class="center-wrap fade-enter"><div>🔎 Checking available CSVs…</div></div>', unsafe_allow_html=True)

    # Determine if new master CSVs are present.  Prefer the new pipeline if
    # master files exist; otherwise fall back to the legacy bn/kn files.
    if drive_enabled():
        # Drive case: list master CSVs first
        m_bn_meta, m_kn_meta = list_master_csvs_drive_root(GDRIVE_FOLDER_ID)
        m_bn_paths = [ensure_csv_cached(m, subdir="root_master/bn") for m in m_bn_meta]
        m_kn_paths = [ensure_csv_cached(m, subdir="root_master/kn") for m in m_kn_meta]
    else:
        m_bn_local, m_kn_local = list_master_csvs_local(ROOT_LOCAL)
        m_bn_paths = [Path(p) for p in m_bn_local]
        m_kn_paths = [Path(p) for p in m_kn_local]

    use_new_master = bool(m_bn_paths or m_kn_paths)

    if use_new_master:
        # New pipeline: build date index from master CSVs
        center.empty(); center = st.empty()
        with center.container():
            st.markdown('<div class="center-wrap"><div>⬇️ Downloading master CSVs…</div></div>', unsafe_allow_html=True)
        # Already cached above via ensure_csv_cached
        center.empty(); center = st.empty()
        with center.container():
            st.markdown('<div class="center-wrap"><div>🧮 Building date index…</div></div>', unsafe_allow_html=True)
        bn_index = build_date_index_from_masters(m_bn_paths) if m_bn_paths else {}
        kn_index = build_date_index_from_masters(m_kn_paths) if m_kn_paths else {}
        center.empty()

        bn_dates = sorted(bn_index.keys())
        kn_dates = sorted(kn_index.keys())
        combined_dates = sorted(set(bn_dates).union(set(kn_dates)))

        src = st.selectbox("Source", ["KōreroNET (kn)", "BirdNET (bn)", "Combined"], index=0)
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01)

        if src == "Combined":
            options, help_txt = combined_dates, "Dates present in any master file."
        elif src == "BirdNET (bn)":
            options, help_txt = bn_dates, "Dates present in BirdNET master files."
        else:
            options, help_txt = kn_dates, "Dates present in KōreroNET master files."

        if not options:
            st.warning(f"No available dates for {src}.")
            st.stop()

        d = calendar_pick(options, "Day", help_txt)

        # Use centered layout for Show results button
        btn_cols = st.columns([1,1,1])
        show = False
        with btn_cols[1]:
            show = st.button("Show results", type="primary", key="tab1_show_btn_new")
        if show:
            with st.spinner("Rendering heatmap…"):
                if src == "BirdNET (bn)":
                    df = load_master_for_day(bn_index.get(d, []), d)
                elif src == "KōreroNET (kn)":
                    df = load_master_for_day(kn_index.get(d, []), d)
                else:
                    df_bn = load_master_for_day(bn_index.get(d, []), d)
                    df_kn = load_master_for_day(kn_index.get(d, []), d)
                    df = pd.concat([df_bn, df_kn], ignore_index=True)
                if df.empty:
                    st.warning("No detections for this day.")
                else:
                    # Standardize to legacy column names for heatmap
                    df2 = df.copy()
                    df2["Confidence"] = pd.to_numeric(df2["Probability"], errors="coerce")
                    df2["ActualTime"] = df2["ActualStartTime"]
                    make_heatmap(df2[["Label","Confidence","ActualTime"]], min_conf, f"{src.split()[0]} • {d.isoformat()}")
    else:
        # Legacy pipeline: fall back to bn*.csv and kn*.csv
        if drive_enabled():
            bn_meta, kn_meta = list_csvs_drive_root(GDRIVE_FOLDER_ID)
        else:
            bn_meta, kn_meta = [], []
        if drive_enabled() and (bn_meta or kn_meta):
            center.empty(); center = st.empty()
            with center.container():
                st.markdown('<div class="center-wrap"><div>⬇️ Downloading root CSVs…</div></div>', unsafe_allow_html=True)
            bn_paths = [ensure_csv_cached(m, subdir="root/bn") for m in bn_meta]
            kn_paths = [ensure_csv_cached(m, subdir="root/kn") for m in kn_meta]
        else:
            bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
            bn_paths = [Path(p) for p in bn_local]
            kn_paths = [Path(p) for p in kn_local]
        center.empty(); center = st.empty()
        with center.container():
            st.markdown('<div class="center-wrap"><div>🧮 Building date index…</div></div>', unsafe_allow_html=True)
        bn_by_date = build_date_index(bn_paths) if bn_paths else {}
        kn_by_date = build_date_index(kn_paths) if kn_paths else {}
        center.empty()
        bn_dates = sorted(bn_by_date.keys())
        kn_dates = sorted(kn_by_date.keys())
        paired_dates = sorted(set(bn_dates).intersection(set(kn_dates)))
        src = st.selectbox("Source", ["KōreroNET (kn)", "BirdNET (bn)", "Combined"], index=0)
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01)
        if src == "Combined":
            options, help_txt = paired_dates, "Only dates that have BOTH BN & KN detections."
        elif src == "BirdNET (bn)":
            options, help_txt = bn_dates, "Dates present in any BN file."
        else:
            options, help_txt = kn_dates, "Dates present in any KN file."
        if not options:
            st.warning(f"No available dates for {src}."); st.stop()
        d = calendar_pick(options, "Day", help_txt)
        def load_and_filter(paths: List[Path], kind: str, day_selected: date):
            frames = []
            for p in paths:
                try:
                    df = load_csv(p)
                    std = standardize_df(df, kind)
                    std = std[std["ActualTime"].dt.date == day_selected]
                    if not std.empty: frames.append(std)
                except Exception:
                    pass
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        # Centered Show results button for legacy path
        btn_cols2 = st.columns([1,1,1])
        show_legacy = False
        with btn_cols2[1]:
            show_legacy = st.button("Show results", type="primary", key="tab1_show_btn_legacy")
        if show_legacy:
            with st.spinner("Rendering heatmap…"):
                if src == "BirdNET (bn)":
                    make_heatmap(load_and_filter(bn_by_date.get(d, []), "bn", d), min_conf, f"BirdNET • {d.isoformat()}")
                elif src == "KōreroNET (kn)":
                    make_heatmap(load_and_filter(kn_by_date.get(d, []), "kn", d), min_conf, f"KōreroNET • {d.isoformat()}")
                else:
                    df_bn = load_and_filter(bn_by_date.get(d, []), "bn", d)
                    df_kn = load_and_filter(kn_by_date.get(d, []), "kn", d)
                    make_heatmap(pd.concat([df_bn, df_kn], ignore_index=True), min_conf, f"Combined (BN+KN) • {d.isoformat()}")

# ================================
# TAB 2 — Verify (snapshot date)
# ================================
with tab_verify:
    if not drive_enabled():
        st.error("Google Drive is not configured in secrets."); st.stop()

    # Display progress indicator
    center2 = st.empty()
    with center2.container():
        st.markdown('<div class="center-wrap fade-enter"><div>📚 Loading master detections…</div></div>', unsafe_allow_html=True)

    # Load old snapshot master index (legacy pipeline)
    master_old = build_master_index_by_snapshot_date(GDRIVE_FOLDER_ID)

    # Load new master CSVs (new pipeline).  These files reside in the
    # "From the node" root.  If none are present, the lists will be empty.
    if drive_enabled():
        n_bn_meta, n_kn_meta = list_master_csvs_drive_root(GDRIVE_FOLDER_ID)
        n_bn_paths = [ensure_csv_cached(m, subdir="root_master/bn") for m in n_bn_meta]
        n_kn_paths = [ensure_csv_cached(m, subdir="root_master/kn") for m in n_kn_meta]
    else:
        n_bn_local, n_kn_local = list_master_csvs_local(ROOT_LOCAL)
        n_bn_paths = [Path(p) for p in n_bn_local]
        n_kn_paths = [Path(p) for p in n_kn_local]

    frames_new: List[pd.DataFrame] = []
    # Build mapping of Kind to drive folder id (only used for new pipeline)
    bk_map = find_backup_audio_folders(GDRIVE_FOLDER_ID) if drive_enabled() else {"BN": "", "KN": ""}
    # Read new master CSVs and annotate with Kind and ChunkDriveFolderId
    for p in n_bn_paths:
        df = _read_master_standard(p, kind="BN")
        if df.empty:
            continue
        df["Date"] = df["ActualStartTime"].dt.date
        df.rename(columns={"Probability": "Confidence"}, inplace=True)
        df["ChunkName"] = df["Clip"]
        df["ChunkDriveFolderId"] = bk_map.get("BN", "")
        frames_new.append(df[["Date", "Kind", "Label", "Confidence", "ChunkName", "ChunkDriveFolderId"]])
    for p in n_kn_paths:
        df = _read_master_standard(p, kind="KN")
        if df.empty:
            continue
        df["Date"] = df["ActualStartTime"].dt.date
        df.rename(columns={"Probability": "Confidence"}, inplace=True)
        df["ChunkName"] = df["Clip"]
        df["ChunkDriveFolderId"] = bk_map.get("KN", "")
        frames_new.append(df[["Date", "Kind", "Label", "Confidence", "ChunkName", "ChunkDriveFolderId"]])
    df_new = pd.concat(frames_new, ignore_index=True) if frames_new else pd.DataFrame(columns=["Date","Kind","Label","Confidence","ChunkName","ChunkDriveFolderId"])

    # Prepare old master DataFrame (if any)
    if not master_old.empty:
        # master_old contains Date, Kind, Label, Confidence, Start, End, WavBase, ChunkName, ChunkDriveFolderId, SnapId, SnapName
        df_old = master_old.copy()
        df_old = df_old[["Date", "Kind", "Label", "Confidence", "ChunkName", "ChunkDriveFolderId"]]
    else:
        df_old = pd.DataFrame(columns=["Date","Kind","Label","Confidence","ChunkName","ChunkDriveFolderId"])

    # Combine new and old detections
    master_all = pd.concat([df_old, df_new], ignore_index=True)
    center2.empty()

    if master_all.empty:
        st.warning("No detections found in any master CSV.")
        st.stop()

    # Filter by user-selected source and confidence
    colA, colB = st.columns([2,1])
    with colA:
        src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0)
    with colB:
        min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01)

    df_pool = master_all.copy()
    df_pool = df_pool[pd.to_numeric(df_pool["Confidence"], errors="coerce") >= float(min_conf_v)]
    if src_mode_v == "KōreroNET (KN)":
        df_pool = df_pool[df_pool["Kind"] == "KN"]
    elif src_mode_v == "BirdNET (BN)":
        df_pool = df_pool[df_pool["Kind"] == "BN"]
    # If after filtering we have no rows, inform the user
    if df_pool.empty:
        st.info("No detections above the selected confidence.")
        st.stop()

    avail_days = sorted(df_pool["Date"].unique())
    day_pick = st.date_input("Day", value=avail_days[-1], min_value=avail_days[0], max_value=avail_days[-1])

    day_df = df_pool[df_pool["Date"] == day_pick]
    if day_df.empty:
        st.warning("No detections for the chosen date.")
        st.stop()

    # Build label list with counts
    counts = day_df.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox(
        "Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_pick.isoformat()}::{src_mode_v}",
    )

    playlist = day_df[day_df["Label"] == species].copy()
    # Sort for deterministic navigation: sort by time or chunk name
    playlist.sort_values(["Kind", "ChunkName"], inplace=True)
    playlist.reset_index(drop=True, inplace=True)

    idx_key = f"v2_idx::{day_pick.isoformat()}::{src_mode_v}::{species}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    idx = st.session_state[idx_key] % len(playlist)

    # Center navigation buttons using columns
    nav_cols = st.columns([2,1,1,1,2])
    autoplay = False
    with nav_cols[1]:
        if st.button("⏮ Prev"):
            idx = (idx - 1) % len(playlist)
            autoplay = True
    with nav_cols[2]:
        if st.button("▶ Play"):
            autoplay = True
    with nav_cols[3]:
        if st.button("⏭ Next"):
            idx = (idx + 1) % len(playlist)
            autoplay = True
    st.session_state[idx_key] = idx

    row = playlist.iloc[idx]
    st.markdown(
        f"**Date:** {row['Date']}  |  **File:** `{row['ChunkName']}`  |  **Kind:** {row['Kind']}  |  **Confidence:** {float(row['Confidence']):.3f}"
    )

    # Unified audio player: use ChunkName, ChunkDriveFolderId and Kind
    def _play_audio_unified(row_: pd.Series, auto: bool):
        chunk_name = str(row_.get("ChunkName", "") or "")
        folder_id = str(row_.get("ChunkDriveFolderId", "") or "")
        kind = str(row_.get("Kind", "")).upper() or "UNK"
        if not chunk_name or not folder_id:
            st.warning("No audio mapping available.")
            return
        # Use kind as subdir name (BN or KN)
        subdir = kind
        with st.spinner("Fetching audio…"):
            cached = ensure_chunk_cached(chunk_name, folder_id, subdir=subdir)
        if not cached or not cached.exists():
            st.warning("Audio chunk not found in Drive backup.")
            return
        try:
            with open(cached, "rb") as f:
                data = f.read()
        except Exception:
            st.warning("Cannot open audio chunk.")
            return
        if auto:
            import base64
            b64 = base64.b64encode(data).decode()
            st.markdown(
                f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>',
                unsafe_allow_html=True,
            )
        else:
            st.audio(data, format="audio/wav")

    _play_audio_unified(row, autoplay)

# =====================
# TAB 3 — Power graph
# =====================
with tab3:
    st.subheader("Node Power History")
    if not drive_enabled():
        st.error("Google Drive is not configured in secrets."); st.stop()

    # Locate 'Power logs' under the Drive root
    def find_subfolder_by_name(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
        kids = list_children(root_id, max_items=2000)
        for k in kids:
            if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name","").lower()==name_ci.lower():
                return k
        return None

    logs_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
    if not logs_folder:
        st.warning("Could not find 'Power logs' folder under the Drive root.")
        st.stop()

    LOG_RE = re.compile(r"^power_history_(\d{8})_(\d{6})\.log$", re.IGNORECASE)

    @st.cache_data(show_spinner=True)
    def list_power_logs(folder_id: str) -> List[Dict[str, Any]]:
        kids = list_children(folder_id, max_items=2000)
        files = [k for k in kids if k.get("mimeType") != "application/vnd.google-apps.folder" and LOG_RE.match(k.get("name",""))]
        files.sort(key=lambda m: m.get("name",""), reverse=True)
        return files

    @st.cache_data(show_spinner=False)
    def ensure_log_cached(meta: Dict[str, Any]) -> Path:
        local_path = POWER_CACHE / meta["name"]
        if not local_path.exists():
            download_to(local_path, meta["id"])
        return local_path

    def _parse_float_list(line: str) -> List[float]:
        try:
            payload = line.split(",", 1)[1]
        except Exception:
            return []
        vals = []
        for tok in payload.strip().split(","):
            tok = tok.strip().rstrip(".")
            try:
                vals.append(float(tok))
            except Exception:
                pass
        return vals

    def parse_power_log(path: Path) -> Optional[pd.DataFrame]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines: return None

        # Header timestamp (first line)
        try:
            head_dt = datetime.strptime(lines[0], "%Y-%m-%d %H:%M:%S")
        except Exception:
            head_dt = None
            for l in lines[:3]:
                m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", l)
                if m:
                    try:
                        head_dt = datetime.strptime(m.group(0), "%Y-%m-%d %H:%M:%S")
                        break
                    except Exception:
                        pass
        if head_dt is None:
            return None

        wh_line   = next((l for l in lines if l.upper().startswith("PH_WH")),   "")
        mah_line  = next((l for l in lines if l.upper().startswith("PH_MAH")),  "")
        soci_line = next((l for l in lines if l.upper().startswith("PH_SOCI")), "")
        socv_line = next((l for l in lines if l.upper().startswith("PH_SOCV")), "")

        WH   = _parse_float_list(wh_line)
        mAh  = _parse_float_list(mah_line)
        SoCi = _parse_float_list(soci_line)
        SoCv = _parse_float_list(socv_line)

        L = max(len(WH), len(mAh), len(SoCi), len(SoCv))
        if L == 0: return None

        def _pad_left(arr: List[float], n: int) -> List[float]:
            return [np.nan]*(n-len(arr)) + arr if len(arr) < n else arr[:n]

        WH   = _pad_left(WH,   L)
        mAh  = _pad_left(mAh,  L)
        SoCi = _pad_left(SoCi, L)
        SoCv = _pad_left(SoCv, L)

        # Build time axis: last element corresponds to head_dt
        times = [head_dt - timedelta(hours=(L-1 - i)) for i in range(L)]
        df = pd.DataFrame({"t": times, "PH_WH": WH, "PH_mAh": mAh, "PH_SoCi": SoCi, "PH_SoCv": SoCv})
        return df

    @st.cache_data(show_spinner=True)
    def build_power_timeseries(folder_id: str, latest_n: int = 7) -> pd.DataFrame:
        files = list_power_logs(folder_id)
        if not files:
            return pd.DataFrame(columns=["t","PH_WH","PH_mAh","PH_SoCi","PH_SoCv"])
        frames: List[pd.DataFrame] = []
        for meta in files[:max(1, latest_n)]:
            local = ensure_log_cached(meta)
            df = parse_power_log(local)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["t","PH_WH","PH_mAh","PH_SoCi","PH_SoCv"])
        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values("t", inplace=True)
        merged = merged.drop_duplicates(subset=["t"], keep="last")
        # At this point merged may contain zeros when the node restarts.  Treat
        # zeros as missing values and interpolate over hourly gaps.  Drop the
        # existing index and set the time column as the index for resampling.
        for c in ["PH_WH", "PH_mAh", "PH_SoCi", "PH_SoCv"]:
            # convert exactly zero values to NaN
            merged.loc[merged[c] == 0.0, c] = np.nan
        # Set time as index and enforce hourly frequency
        merged.set_index("t", inplace=True)
        # Generate continuous hourly index spanning the full range
        merged = merged.asfreq("H")
        # Interpolate missing values using time-based interpolation
        merged = merged.interpolate(method="time", limit_direction="both")
        merged.reset_index(inplace=True)
        return merged

    cols = st.columns([2,1])
    with cols[0]:
        lookback = st.number_input(
            "How many latest logs to stitch",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
            key="tab3_lookback",
        )
    # Reserve second column for spacing
    with cols[1]:
        pass

    # Center the 'Show results' button for better aesthetics
    btn_cols3 = st.columns([1,1,1])
    show_power = False
    with btn_cols3[1]:
        show_power = st.button("Show results", type="primary", key="tab3_show_btn")

    if show_power:
        with st.spinner("Building power time-series…"):
            ts = build_power_timeseries(logs_folder["id"], latest_n=int(lookback))

        if ts.empty:
            st.warning("No parsable power logs found.")
        else:
            fig = go.Figure()
            # SoC_i (%) on y1
            fig.add_trace(
                go.Scatter(x=ts["t"], y=ts["PH_SoCi"], mode="lines", name="SoC_i (%)", yaxis="y1")
            )
            # Wh on y2
            fig.add_trace(
                go.Scatter(x=ts["t"], y=ts["PH_WH"], mode="lines", name="Energy (Wh)", yaxis="y2")
            )
            fig.update_layout(
                title="Power / State of Charge over time",
                xaxis=dict(title="Time"),
                yaxis=dict(title="SoC (%)", range=[0, 100]),
                yaxis2=dict(title="Wh", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            # Quick last-point indicators
            last = ts.iloc[-1]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Last SoC_i (%)", f"{last['PH_SoCi']:.1f}")
            with c2:
                st.metric("Last Energy (Wh)", f"{last['PH_WH']:.2f}")
