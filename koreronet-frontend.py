#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KōreroNET Dashboard (with splash + Drive)
# ------------------------------------------------------------
# - Splash screen ("KōreroNET" + "AUT") for ~2s, then main UI
# - NEW: Welcome overlay summarising latest detections (top-3) before tabs render
# - Tab 1: Root CSV heatmaps (Drive or local) with calendar + min confidence
# - Tab 2: Verify using snapshot date (Backup/YYYYMMDD_HHMMSS);
#          on-demand audio chunk fetch from Drive (no full directory downloads).
# - Tab 3: Power graph from "Power logs" (Drive), stitches latest N logs;
#          dual y-axes: SoC_i (%) and Wh.
#
# Streamlit secrets required (only if OFFLINE_DEPLOY=False):
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

import os, io, re, glob, json, time, uuid
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import functools
import random

try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None  # graceful fallback

# ★ REVISED: added RerunException/StopException guard so _retry never
#   swallows Streamlit's internal control-flow exceptions (was causing
#   silent infinite loops and hangs).
def _retry(exceptions=(Exception,), tries=3, base_delay=0.35, max_delay=1.2, jitter=0.25):
    """Simple retry decorator with jittered backoff."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            _tries = tries
            _delay = base_delay
            while _tries > 0:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    # ★ Never swallow Streamlit control-flow exceptions
                    if type(e).__name__ in ("RerunException", "StopException", "RerunData"):
                        raise
                    _tries -= 1
                    if _tries <= 0:
                        raise
                    time.sleep(_delay + random.random() * jitter)
                    _delay = min(max_delay, _delay * 1.6)
        return wrap
    return deco


# ─────────────────────────────────────────────────────────────
# Page config + theme-adaptive CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="KōreroNET Dashboard", page_icon="🐦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --kn-green: #2E7D32;
  --kn-green-l: #4CAF50;
  --kn-radius: 10px;
  --kn-font: 'DM Sans', system-ui, sans-serif;
  --kn-mono: 'JetBrains Mono', ui-monospace, monospace;
}

html, body, [data-testid="stAppViewContainer"] { font-family: var(--kn-font) !important; }
.block-container { padding-top: 1.2rem; max-width: 1320px; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
code, .stCode, [data-testid="stCode"] { font-family: var(--kn-mono) !important; }

/* ── Theme-adaptive stat cards ── */
.kn-metrics { display: flex; gap: .75rem; flex-wrap: wrap; margin: .75rem 0; }
.kn-metric {
  flex: 1; min-width: 130px; padding: .8rem 1rem;
  border-radius: var(--kn-radius); text-align: center;
  border: 1px solid rgba(128,128,128,.15);
}
.kn-metric-val { font-size: 1.5rem; font-weight: 800; }
.kn-metric-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; opacity: .65; margin-top: .15rem; }

/* Light mode */
@media (prefers-color-scheme: light) {
  .kn-metric { background: #ffffff; border-color: rgba(0,0,0,.08); }
  .kn-metric-val { color: #1B5E20; }
}
/* Dark mode */
@media (prefers-color-scheme: dark) {
  .kn-metric { background: #1E1E1E; border-color: rgba(255,255,255,.08); }
  .kn-metric-val { color: #4CAF50; }
}

/* ── Player card ── */
.kn-player {
  border: 1px solid rgba(128,128,128,.15); border-radius: var(--kn-radius);
  padding: 1rem 1.25rem; margin: .75rem 0;
}
.kn-player-meta { font-family: var(--kn-mono); font-size: .82rem; line-height: 1.6; }

/* ── Features grid ── */
.features-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr));
  gap: .65rem; margin: .75rem 0;
}
.feature-card {
  display: flex; align-items: flex-start; gap: .55rem;
  border: 1px solid rgba(128,128,128,.1); border-radius: 8px; padding: .6rem .8rem;
}
.feature-icon { font-size: 1.2rem; }
.feature-text { font-size: .88rem; line-height: 1.35; opacity: .8; }

/* ── Overlay pills ── */
.overlay-pill {
  display: inline-block; padding: .3rem .65rem; border-radius: 999px;
  font-size: .82rem; border: 1px solid rgba(128,128,128,.15);
  margin: .2rem .2rem 0 0;
}
.overlay-sub { font-size: 1rem; opacity: .85; margin-bottom: .5rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Constants & Regex
# ============================================================================
CHUNK_RE = re.compile(
    r"^(?P<root>\d{8}_\d{6})__(?P<tag>bn|kn)_(?P<s>\d+\.\d{2})_(?P<e>\d+\.\d{2})__(?P<label>.+?)__p(?P<conf>\d+\.\d{2})\.wav$",
    re.IGNORECASE,
)
NEW_ROOT_BN = re.compile(r"^\d{8}_\d{6}_birdnet_master\.csv$", re.IGNORECASE)
NEW_ROOT_KN = re.compile(r"^\d{8}_\d{6}_koreronet_master\.csv$", re.IGNORECASE)
SNAP_RE     = re.compile(r"^(\d{8})_(\d{6})$", re.IGNORECASE)
LOG_AUTOSTART_RE = re.compile(r"^(\d{8})_(\d{6})__gui_autostart\.log$", re.IGNORECASE)
CUTOFF_NEW  = date(2025, 10, 31)  # new format becomes active on/after this date

# ★ REVISED: only fetch the fields we actually use (was fetching ~50 fields)
DRIVE_FIELDS = "nextPageToken, files(id,name,mimeType,modifiedTime,size)"

# Preset field nodes (extend as you add sites)
NODES = [
    {
        "key": "Auckland-Sunnyhills",
        "name": "Auckland — Sunnyhills",
        "lat": -36.9003,
        "lon": 174.8839,
        "desc": "Field node in Sunnyhills, Auckland",
    },
    # Add more nodes here as needed...
]
NODE_KEYS = [n["key"] for n in NODES]



# ============================================================================
# Utility: unique keys
# ============================================================================
if "_sess_salt" not in st.session_state:
    st.session_state["_sess_salt"] = str(uuid.uuid4())[:8]

def k(name: str) -> str:
    return f"{name}::{st.session_state['_sess_salt']}"


# ============================================================================
if "_sess_salt" not in st.session_state:
    st.session_state["_sess_salt"] = str(uuid.uuid4())[:8]

def k(name: str) -> str:
    """Unique widget key helper."""
    return f"{name}::{st.session_state['_sess_salt']}"

# ============================================================================
# Caches & local fallback
# ============================================================================
CACHE_ROOT   = Path("/tmp/koreronet_cache")
CSV_CACHE    = CACHE_ROOT / "csv"
CHUNK_CACHE  = CACHE_ROOT / "chunks"
POWER_CACHE  = CACHE_ROOT / "power"
for _p in (CSV_CACHE, CHUNK_CACHE, POWER_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

DEFAULT_ROOT = r"G:\My Drive\From the node"
ROOT_LOCAL   = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)

# ---- Offline/Online switch --------------------------------------------------
OFFLINE_DEPLOY: bool = False   # ← flip to False when you want online mode

# Your local Google Drive clone root:
ROOT_LOCAL = os.getenv("KORERONET_DATA_ROOT", r"G:\My Drive\From the node")

def _secret_or_env(name: str, default=None):
    v = os.getenv(name)
    if v is not None:
        return v
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        return default

# If offline, map "Drive root id" to local clone path; otherwise read from env/secrets.
GDRIVE_FOLDER_ID = ROOT_LOCAL if OFFLINE_DEPLOY else _secret_or_env("GDRIVE_FOLDER_ID", None)


# ============================================================================
# Secrets / Drive
# ============================================================================

def _normalize_private_key(pk: str) -> str:
    if not isinstance(pk, str): return pk
    if "\\n" in pk: pk = pk.replace("\\n", "\n")
    if "-----BEGIN PRIVATE KEY-----" in pk and "-----END PRIVATE KEY-----" in pk:
        if "-----BEGIN PRIVATE KEY-----\n" not in pk:
            pk = pk.replace("-----BEGIN PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----\n", 1)
        if "\n-----END PRIVATE KEY-----" not in pk:
            pk = pk.replace("-----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----", 1)
    return pk

# ★ REVISED: use @st.cache_resource so Drive client is built once per app
#   lifecycle, not per session. This is the single biggest auth speedup.
@st.cache_resource(show_spinner=False)
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
    # ★ REVISED: no more session_state storage; cache_resource handles it
    if OFFLINE_DEPLOY:
        return "offline"
    return _build_drive_client()

def drive_enabled() -> bool:
    return bool(GDRIVE_FOLDER_ID and get_drive_client())

# ============================================================================
# Drive helpers + epoch (freshness)
# ============================================================================

# ★ FIX: cache list_children — was making 426+ uncached API calls per cold load
@st.cache_data(ttl=300, max_entries=500, show_spinner=False)
@_retry()
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    drive = get_drive_client()
    if not drive: return []
    items, token = [], None
    while True:
        page_size = min(100, max_items - len(items))
        if page_size <= 0: break
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields=DRIVE_FIELDS,  # ★ was fetching ~50 fields including md5Checksum, parents
            pageSize=page_size, pageToken=token, orderBy="folder,name_natural",
            includeItemsFromAllDrives=True, supportsAllDrives=True, corpora="allDrives",
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token: break
    return items

@_retry()
def download_to(path: Path, file_id: str, force: bool = False) -> Path:
    drive = get_drive_client()
    from googleapiclient.http import MediaIoBaseDownload
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not force) and path.exists() and path.stat().st_size > 0:  # ★ also check non-zero
        return path
    req = drive.files().get_media(fileId=file_id)
    with open(path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return path

# ★ REVISED: added ttl + max_entries
@st.cache_data(ttl=180, max_entries=10, show_spinner=False)
@_retry()
def _compute_drive_epoch(root_id: str, nocache: str = "stable") -> str:
    drive = get_drive_client()
    if not drive: return "no-drive"
    def _max_mtime(kids: List[Dict[str, Any]]) -> str:
        if not kids: return ""
        return max((k.get("modifiedTime","") for k in kids), default="")
    root_kids = list_children(root_id, max_items=2000)
    root_max  = _max_mtime(root_kids)
    bkid = None
    for k in root_kids:
        if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name","").lower() == "backup":
            bkid = k.get("id")
            break
    back_max = ""
    if bkid:
        back_kids = list_children(bkid, max_items=2000)
        back_max  = _max_mtime(back_kids)
    token = "|".join([root_max, back_max])
    return token or time.strftime("%Y%m%d%H%M%S")

def _ensure_epoch_key():
    if not drive_enabled():
        return
    try:
        new_epoch = _compute_drive_epoch(GDRIVE_FOLDER_ID, nocache="stable")
    except Exception as e:
        if type(e).__name__ in ("RerunException", "StopException", "RerunData"):
            raise
        st.sidebar.warning("Drive indexing hiccup (using last known state).")
        return
    old_epoch = st.session_state.get("DRIVE_EPOCH")
    if old_epoch is None:
        st.session_state["DRIVE_EPOCH"] = new_epoch
    elif new_epoch != old_epoch:
        st.cache_data.clear()
        st.session_state["DRIVE_EPOCH"] = new_epoch


_ensure_epoch_key()

def _folder_children_cached(folder_id: str) -> List[Dict[str, Any]]:
    epoch = st.session_state.get("DRIVE_EPOCH", "0")
    key = f"drive_kids::{epoch}::{folder_id}"
    if key not in st.session_state:
        st.session_state[key] = list_children(folder_id, max_items=2000)
    return st.session_state[key]


def ensure_csv_cached(meta: Dict[str, Any], subdir: str, cache_epoch: str = "", force: bool = False) -> Path:
    local_path = (CSV_CACHE / subdir / meta["name"])
    return download_to(local_path, meta["id"], force=force)

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str, force: bool = False) -> Optional[Path]:
    local_path = CHUNK_CACHE / subdir / chunk_name
    if (not force) and local_path.exists() and local_path.stat().st_size > 0:  # ★ non-zero check
        return local_path

    kids = _folder_children_cached(folder_id)
    name_to_id = {k.get("name"): k.get("id") for k in kids}
    if chunk_name in name_to_id:
        try:
            download_to(local_path, name_to_id[chunk_name], force=force)
            return local_path
        except Exception:
            return None

    target = _parse_chunk_filename(chunk_name)
    if not target:
        return None

    root_t  = target["root"]
    tag_t   = target["tag"]
    s_t     = target["s"]; e_t = target["e"]
    label_t = target["label"].lower()

    candidates = []
    for k in kids:
        nm = k.get("name","")
        if not nm.lower().endswith(".wav"):
            continue
        info = _parse_chunk_filename(nm)
        if not info: continue
        if info["root"] == root_t and info["tag"] == tag_t and info["label"].lower() == label_t:
            candidates.append((info, k.get("id"), nm))
    if not candidates:
        return None

    tol = 0.75
    def score(cinfo):
        s_c, e_c = cinfo["s"], cinfo["e"]
        contains = (s_c - tol) <= s_t and (e_c + tol) >= e_t
        cen_diff = abs(((s_c + e_c) * 0.5) - ((s_t + e_t) * 0.5))
        return (0 if contains else 1, cen_diff)

    candidates.sort(key=lambda it: score(it[0]))
    best_info, best_id, best_name = candidates[0]
    try:
        download_to(local_path, best_id, force=force)
        st.caption(f"⚠️ Used fuzzy match: requested `{chunk_name}` → found `{best_name}`")
        return local_path
    except Exception:
        return None

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

# ============================================================================
# Tab 1 helpers — list/standardize across old & new root CSVs
# ============================================================================
# ★ REVISED: added ttl + max_entries to ALL cache decorators
@st.cache_data(ttl=600, max_entries=20, show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    bn_paths += sorted(glob.glob(os.path.join(root, "[0-9]"*8 + "_" + "[0-9]"*6 + "_birdnet_master.csv")))
    kn_paths += sorted(glob.glob(os.path.join(root, "[0-9]"*8 + "_" + "[0-9]"*6 + "_koreronet_master.csv")))
    return bn_paths, kn_paths

@st.cache_data(ttl=300, max_entries=20, show_spinner=False)
def list_csvs_drive_root(folder_id: str, cache_epoch: str, nocache: str = "stable") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kids = list_children(folder_id, max_items=2000)
    bn, kn = [], []
    for k in kids:
        n = k.get("name","")
        nl = n.lower()
        if nl.endswith(".csv"):
            if nl.startswith("bn") or NEW_ROOT_BN.match(n):
                bn.append(k)
            elif nl.startswith("kn") or NEW_ROOT_KN.match(n):
                kn.append(k)
    bn.sort(key=lambda m: m.get("name","")); kn.sort(key=lambda m: m.get("name",""))
    return bn, kn

def _std_newformat(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Label"]       = df.get("Label", "Unknown")
    out["Confidence"]  = pd.to_numeric(df.get("Probability", np.nan), errors="coerce")
    out["ActualTime"]  = pd.to_datetime(df.get("ActualStartTime", pd.NaT), errors="coerce")
    return out.dropna(subset=["Label","ActualTime"])

def _std_legacy(df: pd.DataFrame, kind: str) -> pd.DataFrame:
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

def standardize_root_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    cols = set(c.lower() for c in df.columns)
    if {"clip","actualstarttime","label","probability"} <= cols:
        return _std_newformat(df)
    return _std_legacy(df, kind)

# ★ REVISED: added ttl + max_entries; use usecols for faster scanning
@st.cache_data(ttl=600, max_entries=40, show_spinner=False)
def extract_dates_from_csv(path: str | Path) -> List[date]:
    path = str(path)
    dates = set()
    try:
        for chunk in pd.read_csv(path, chunksize=5000,
                                  usecols=lambda c: c in ("ActualStartTime", "ActualTime")):
            if "ActualStartTime" in chunk.columns:
                s = pd.to_datetime(chunk["ActualStartTime"], errors="coerce")
            else:
                if "ActualTime" not in chunk.columns: continue
                s = pd.to_datetime(chunk["ActualTime"], errors="coerce", dayfirst=True)
            dates.update(ts.date() for ts in s.dropna())
    except Exception:
        return []
    return sorted(dates)

@st.cache_data(ttl=600, max_entries=20, show_spinner=False)
def build_date_index(paths: List[Path], kind: str) -> Dict[date, List[str]]:
    idx: Dict[date, List[str]] = {}
    for p in paths:

        ds = extract_dates_from_csv(p)
        for d in ds:
            idx.setdefault(d, []).append(str(p))
    return idx

@st.cache_data(ttl=300, max_entries=30, show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(str(path))

# ============================================================================
# Snapshot/master logic for Tab 2
# ============================================================================
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

# ★ FIX: accept pre-fetched kids to avoid redundant list_children calls per snapshot
def _find_chunk_dirs(snapshot_id: str, kids: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    if kids is None:
        kids = list_children(snapshot_id, max_items=2000)
    kn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "koreronet" in k.get("name","").lower()]
    bn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "birdnet"   in k.get("name","").lower()]
    return {"KN": (kn[0]["id"] if kn else snapshot_id), "BN": (bn[0]["id"] if bn else snapshot_id)}

# ★ FIX: accept pre-fetched kids to avoid redundant list_children calls
def _find_named_in_snapshot(snapshot_id: str, exact_name: str, kids: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if kids is None:
        kids = list_children(snapshot_id, max_items=2000)
    files_only = [f for f in kids if f.get("mimeType") != "application/vnd.google-apps.folder"]
    for f in files_only:
        if f.get("name","").lower() == exact_name.lower():
            return f
    subfolders = [f for f in kids if f.get("mimeType") == "application/vnd.google-apps.folder"]
    for sf in subfolders:
        sub_files = list_children(sf["id"], max_items=2000)
        for f in sub_files:
            if f.get("mimeType") != "application/vnd.google-apps.folder" and f.get("name","").lower() == exact_name.lower():
                return f
    return None

def _match_legacy_master_name(name: str, kind: str) -> bool:
    n = name.lower()
    if kind == "KN":
        return (("koreronet" in n) and ("detect" in n) and n.endswith(".csv"))
    else:
        return (("birdnet" in n) and ("detect" in n) and n.endswith(".csv"))

# ★ FIX: fast month listing — 1 API call, no CSV parsing
@st.cache_data(ttl=300, max_entries=5, show_spinner=False)
def _list_snapshot_months(root_folder_id: str, _epoch: str = "") -> List[Tuple[int, int]]:
    """List available (year, month) pairs from Backup/ folder names. Fast: 1 API call."""
    backup = _find_backup_folder(root_folder_id)
    if not backup:
        return []
    snaps = [kid for kid in list_children(backup["id"], max_items=2000)
             if kid.get("mimeType") == "application/vnd.google-apps.folder"
             and SNAP_RE.match(kid.get("name", ""))]
    months = set()
    for sn in snaps:
        d = _parse_date_from_snapname(sn["name"])
        if d:
            months.add((d.year, d.month))
    return sorted(months, reverse=True)

# ★ REVISED: added ttl + max_entries + date filtering
@st.cache_data(ttl=300, max_entries=5, show_spinner=True)
def build_master_index_by_snapshot_date(root_folder_id: str, cache_epoch: str, nocache: str = "stable",
                                         date_from: Optional[date] = None, date_to: Optional[date] = None) -> pd.DataFrame:
    backup = _find_backup_folder(root_folder_id)
    if not backup:
        return pd.DataFrame(columns=[
            "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
            "ChunkDriveFolderId","SnapId","SnapName"
        ])

    snaps = [k for k in list_children(backup["id"], max_items=2000)
             if k.get("mimeType")=="application/vnd.google-apps.folder" and SNAP_RE.match(k.get("name",""))]
    snaps.sort(key=lambda m: m.get("name",""), reverse=True)

    frames: List[pd.DataFrame] = []  # ★ FIX: collect DataFrames, not row dicts
    for sn in snaps:
        snap_id, snap_name = sn["id"], sn["name"]
        snap_date = _parse_date_from_snapname(snap_name)
        if not snap_date: continue

        # ★ FIX: skip snapshots outside the requested date range (index only selected month)
        if date_from and snap_date < date_from:
            continue
        if date_to and snap_date > date_to:
            continue

        # ★ FIX: single list_children call per snapshot (was 3+ calls)
        snap_kids = list_children(snap_id, max_items=2000)
        chunk_dirs = _find_chunk_dirs(snap_id, kids=snap_kids)

        if snap_date >= CUTOFF_NEW:
            kn_meta = _find_named_in_snapshot(snap_id, "koreronet_master.csv", kids=snap_kids)
            if kn_meta:
                kn_csv = ensure_csv_cached(kn_meta, subdir=f"snap_{snap_id}/koreronet", cache_epoch=cache_epoch, force=False)
                try:
                    df = pd.read_csv(kn_csv)
                    # ★ FIX: vectorized — was iterrows() over 41K rows
                    df = df.assign(
                        Date=snap_date, Kind="KN",
                        Label=df.get("Label", "Unknown").astype(str),
                        Confidence=pd.to_numeric(df.get("Probability", np.nan), errors="coerce"),
                        Start=np.nan, End=np.nan, WavBase="",
                        ChunkName=df.get("Clip", "").astype(str).str.strip(),
                        ChunkDriveFolderId=chunk_dirs["KN"],
                        SnapId=snap_id, SnapName=snap_name,
                    )[["Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName","ChunkDriveFolderId","SnapId","SnapName"]]
                    frames.append(df)
                except Exception:
                    pass

            bn_meta = _find_named_in_snapshot(snap_id, "birdnet_master.csv", kids=snap_kids)
            if bn_meta:
                bn_csv = ensure_csv_cached(bn_meta, subdir=f"snap_{snap_id}/birdnet", cache_epoch=cache_epoch, force=False)
                try:
                    df = pd.read_csv(bn_csv)
                    label_col = "Label" if "Label" in df.columns else ("Common name" if "Common name" in df.columns else None)
                    conf_col  = "Probability" if "Probability" in df.columns else ("Confidence" if "Confidence" in df.columns else None)
                    df = df.assign(
                        Date=snap_date, Kind="BN",
                        Label=df[label_col].astype(str) if label_col else "Unknown",
                        Confidence=pd.to_numeric(df[conf_col], errors="coerce") if conf_col else np.nan,
                        Start=np.nan, End=np.nan, WavBase="",
                        ChunkName=df.get("Clip", "").astype(str).str.strip(),
                        ChunkDriveFolderId=chunk_dirs["BN"],
                        SnapId=snap_id, SnapName=snap_name,
                    )[["Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName","ChunkDriveFolderId","SnapId","SnapName"]]
                    frames.append(df)
                except Exception:
                    pass
        else:
            # Legacy format — reuse snap_kids (was calling list_children again)
            files_only = [f for f in snap_kids if f.get("mimeType") != "application/vnd.google-apps.folder"]

            kn_legacy = [f for f in files_only if _match_legacy_master_name(f.get("name",""), "KN")]
            if not kn_legacy:
                for sf in [f for f in snap_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]:
                    cand = [f for f in list_children(sf["id"], max_items=2000)
                            if f.get("mimeType") != "application/vnd.google-apps.folder"
                            and _match_legacy_master_name(f.get("name", ""), "KN")]
                    if cand:
                        kn_legacy = cand
                        break
            if kn_legacy:
                kn_legacy.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
                meta = kn_legacy[0]
                kn_csv = ensure_csv_cached(meta, subdir=f"snap_{snap_id}/koreronet", cache_epoch=cache_epoch, force=False)
                try:
                    df = pd.read_csv(kn_csv)
                    # ★ FIX: vectorized legacy KN
                    wav_col = df.get("File", pd.Series(dtype=str)).astype(str).apply(os.path.basename)
                    s_col = pd.to_numeric(df.get("Start", df.get("Start (s)", np.nan)), errors="coerce")
                    e_col = pd.to_numeric(df.get("End",   df.get("End (s)",   np.nan)), errors="coerce")
                    lab_col = df.get("Label", "Unknown").astype(str)
                    conf_col = pd.to_numeric(df.get("Confidence", np.nan), errors="coerce")
                    chunk_names = [_compose_chunk_name("kn", w, s, e, l, c) for w, s, e, l, c
                                   in zip(wav_col, s_col, e_col, lab_col, conf_col)]
                    df = pd.DataFrame({
                        "Date": snap_date, "Kind": "KN", "Label": lab_col,
                        "Confidence": conf_col, "Start": s_col, "End": e_col,
                        "WavBase": wav_col, "ChunkName": chunk_names,
                        "ChunkDriveFolderId": chunk_dirs["KN"],
                        "SnapId": snap_id, "SnapName": snap_name,
                    })
                    frames.append(df)
                except Exception:
                    pass


            bn_legacy = [f for f in files_only if _match_legacy_master_name(f.get("name",""), "BN")]
            if not bn_legacy:
                for sf in [f for f in snap_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]:
                    cand = [f for f in list_children(sf["id"], max_items=2000)
                            if f.get("mimeType") != "application/vnd.google-apps.folder"
                            and _match_legacy_master_name(f.get("name",""), "BN")]
                    if cand: bn_legacy = cand; break
            if bn_legacy:
                bn_legacy.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
                meta = bn_legacy[0]
                bn_csv = ensure_csv_cached(meta, subdir=f"snap_{snap_id}/birdnet", cache_epoch=cache_epoch, force=False)
                try:
                    df = pd.read_csv(bn_csv)
                    # ★ FIX: vectorized legacy BN
                    wav_col = df.get("File", pd.Series(dtype=str)).astype(str).apply(os.path.basename)
                    s_col = pd.to_numeric(df.get("Start (s)", df.get("Start", np.nan)), errors="coerce")
                    e_col = pd.to_numeric(df.get("End (s)",   df.get("End",   np.nan)), errors="coerce")
                    lab_col = df.get("Common name", df.get("Label", "Unknown")).astype(str)
                    conf_col = pd.to_numeric(df.get("Confidence", np.nan), errors="coerce")
                    chunk_names = [_compose_chunk_name("bn", w, s, e, l, c) for w, s, e, l, c
                                   in zip(wav_col, s_col, e_col, lab_col, conf_col)]
                    df = pd.DataFrame({
                        "Date": snap_date, "Kind": "BN", "Label": lab_col,
                        "Confidence": conf_col, "Start": s_col, "End": e_col,
                        "WavBase": wav_col, "ChunkName": chunk_names,
                        "ChunkDriveFolderId": chunk_dirs["BN"],
                        "SnapId": snap_id, "SnapName": snap_name,
                    })
                    frames.append(df)
                except Exception:
                    pass

    if not frames:
        return pd.DataFrame(columns=[
            "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
            "ChunkDriveFolderId","SnapId","SnapName"
        ])
    out = pd.concat(frames, ignore_index=True)  # ★ FIX: concat DataFrames, not build from row dicts
    out.sort_values(["Date","Kind","Label"], ascending=[False, True, True], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

# ============================================================================
# Offline overrides — emulate Google Drive using local filesystem
# ============================================================================
if OFFLINE_DEPLOY:
    GDRIVE_FOLDER_ID = ROOT_LOCAL  # type: ignore

    import datetime as _dt
    from typing import Iterable as _Iterable

    def _mime_for_path(_p: Path) -> str:
        return (
            "application/vnd.google-apps.folder" if _p.is_dir() else "application/octet-stream"
        )

    def drive_enabled() -> bool:  # override
        return True

    def get_drive_client():  # override
        return "offline"

    def list_children(folder_id: str, max_items: int = 2000):  # override
        base = Path(folder_id)
        items = []
        if not base.exists() or not base.is_dir():
            return []
        for i, entry in enumerate(base.iterdir()):
            if i >= max_items:
                break
            stat = entry.stat()
            mt = _dt.datetime.fromtimestamp(stat.st_mtime).isoformat()
            items.append({
                "id": str(entry),
                "name": entry.name,
                "mimeType": _mime_for_path(entry),
                "modifiedTime": mt,
                "size": str(stat.st_size),
                "md5Checksum": None,
                "parents": [str(base)],
            })
        def _key(m):
            return (0 if m.get("mimeType")=="application/vnd.google-apps.folder" else 1, m.get("name",""))
        return sorted(items, key=_key)

    def download_to(path: Path, file_id: str, force: bool = False) -> Path:  # override
        src = Path(file_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if (not force) and path.exists():
            return path
        if src.exists() and src.is_file():
            import shutil as _shutil
            _shutil.copyfile(src, path)
            return path
        return path

    def _folder_children_cached(folder_id: str):
        epoch = st.session_state.get("DRIVE_EPOCH", "0")
        key = f"drive_kids::{epoch}::{folder_id}"
        if key not in st.session_state:
            st.session_state[key] = list_children(folder_id, max_items=2000)
        return st.session_state[key]

    def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str, force: bool = False):
        local_path = CHUNK_CACHE / subdir / chunk_name
        if (not force) and local_path.exists():
            return local_path
        folder = Path(folder_id)
        try:
            exact = folder / chunk_name
            if exact.exists():
                import shutil as _shutil
                local_path.parent.mkdir(parents=True, exist_ok=True)
                _shutil.copyfile(exact, local_path)
                return local_path
        except Exception:
            pass

        target = _parse_chunk_filename(chunk_name)
        if not target or (not folder.exists()):
            return None
        root_t, tag_t = target["root"], target["tag"]
        s_t, e_t = target["s"], target["e"]
        label_t = target["label"].lower()

        candidates = []
        for p in folder.glob("*.wav"):
            nm = p.name
            info = _parse_chunk_filename(nm)
            if not info:
                continue
            if info["root"] == root_t and info["tag"] == tag_t and info["label"].lower() == label_t:
                candidates.append((info, nm))
        if not candidates:
            return None

        tol = 0.75
        def score(cinfo):
            s_c, e_c = cinfo["s"], cinfo["e"]
            contains = (s_c - tol) <= s_t and (e_c + tol) >= e_t
            cen_diff = abs(((s_c + e_c) * 0.5) - ((s_t + e_t) * 0.5))
            return (0 if contains else 1, cen_diff)

        candidates.sort(key=lambda it: score(it[0]))
        best_name = candidates[0][1]
        try:
            import shutil as _shutil
            local_path.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copyfile(folder / best_name, local_path)
            st.caption(f"⚠️ Used fuzzy match: requested `{chunk_name}` → found `{best_name}`")
            return local_path
        except Exception:
            return None

    def find_subfolder_by_name(root_id: str, name_ci: str):
        kids = list_children(root_id, max_items=2000)
        for k in kids:
            if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name", "").lower() == name_ci.lower():
                return k
        return None

    # ★ REVISED: added ttl + max_entries
    @st.cache_data(ttl=300, max_entries=5, show_spinner=True)
    def build_master_index_by_snapshot_date(root_folder_id: str, cache_epoch: str, nocache: str = "stable"):
        backup_path = Path(root_folder_id) / "Backup"
        if not backup_path.exists():
            return pd.DataFrame(columns=[
                "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
                "ChunkDriveFolderId","SnapId","SnapName"
            ])

        snaps = [p for p in backup_path.iterdir() if p.is_dir() and SNAP_RE.match(p.name)]
        snaps.sort(key=lambda p: p.name, reverse=True)

        rows = []
        for sn in snaps:
            snap_name = sn.name
            m = SNAP_RE.match(snap_name)
            if not m:
                continue
            try:
                snap_date = _dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            except Exception:
                continue

            kn_dir = next((p for p in sn.iterdir() if p.is_dir() and "koreronet" in p.name.lower()), sn)
            bn_dir = next((p for p in sn.iterdir() if p.is_dir() and "birdnet" in p.name.lower()), sn)

            if snap_date >= CUTOFF_NEW:
                kn_csv = sn / "koreronet_master.csv"
                if kn_csv.exists():
                    try:
                        df = pd.read_csv(kn_csv)
                        for _, r in df.iterrows():
                            rows.append({
                                "Date": snap_date, "Kind": "KN",
                                "Label": str(r.get("Label","Unknown")),
                                "Confidence": float(r.get("Probability", np.nan)),
                                "Start": np.nan, "End": np.nan, "WavBase": "",
                                "ChunkName": str(r.get("Clip","")),
                                "ChunkDriveFolderId": str(kn_dir),
                                "SnapId": str(sn), "SnapName": snap_name,
                            })
                    except Exception:
                        pass
                bn_csv = sn / "birdnet_master.csv"
                if bn_csv.exists():
                    try:
                        df = pd.read_csv(bn_csv)
                        for _, r in df.iterrows():
                            rows.append({
                                "Date": snap_date, "Kind": "BN",
                                "Label": str(r.get("Label", r.get("Common name","Unknown"))),
                                "Confidence": float(r.get("Probability", r.get("Confidence", np.nan))),
                                "Start": np.nan, "End": np.nan, "WavBase": "",
                                "ChunkName": str(r.get("Clip","")),
                                "ChunkDriveFolderId": str(bn_dir),
                                "SnapId": str(sn), "SnapName": snap_name,
                            })
                    except Exception:
                        pass
            else:
                def _find_legacy(kind: str):
                    pat = "koreronet" if kind=="KN" else "birdnet"
                    cands = [p for p in sn.iterdir() if p.is_file() and (pat in p.name.lower()) and ("detect" in p.name.lower()) and p.suffix.lower()==".csv"]
                    if not cands:
                        for sub in (p for p in sn.iterdir() if p.is_dir()):
                            cands = [p for p in sub.iterdir() if p.is_file() and (pat in p.name.lower()) and ("detect" in p.name.lower()) and p.suffix.lower()==".csv"]
                            if cands:
                                break
                    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[:1]

                kn_legacy = _find_legacy("KN")

                if kn_legacy:
                    try:
                        df = pd.read_csv(kn_legacy[0])
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
                                "ChunkDriveFolderId": str(kn_dir),
                                "SnapId": str(sn), "SnapName": snap_name,
                            })
                    except Exception:
                        pass

                bn_legacy = _find_legacy("BN")
                if bn_legacy:
                    try:
                        df = pd.read_csv(bn_legacy[0])
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
                                "ChunkDriveFolderId": str(bn_dir),
                                "SnapId": str(sn), "SnapName": snap_name,
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

# ============================================================================
# Welcome overlay (latest detections top-3 sentence)
# ============================================================================

def _human_day(d: date) -> str:
    today = datetime.now().date()
    if d == today: return "Today"
    if d == (today - timedelta(days=1)): return "Yesterday"
    return d.strftime("%A, %d %b %Y")

def _safe_plural(n: int, noun: str) -> str:
    return f"{n:,} {noun}"

def _join_top(items: List[Tuple[str, int]]) -> str:
    items = [(lbl, cnt) for lbl, cnt in items if cnt > 0]
    if not items: return ""
    parts = [f"**{_safe_plural(cnt, lbl)}**" for (lbl, cnt) in items[:3]]
    if len(parts) == 1: return parts[0]
    if len(parts) == 2: return " and ".join(parts)
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


# ★ REVISED: added ttl + max_entries
@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def list_autostart_logs(raw_folder_id: str, cache_epoch: str, nocache: str = "stable") -> List[Dict[str, Any]]:
    """Return files under /Power logs/raw matching *__gui_autostart.log, newest first."""
    kids = list_children(raw_folder_id, max_items=2000)
    files = [k for k in kids
             if k.get("mimeType") != "application/vnd.google-apps.folder"
             and LOG_AUTOSTART_RE.match(k.get("name",""))]
    files.sort(key=lambda m: (m.get("name",""), m.get("modifiedTime","")), reverse=True)
    return files

@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def ensure_raw_cached(meta: Dict[str, Any], force: bool = False) -> Path:
    """Download a raw log file to the POWER_CACHE folder."""
    local_path = POWER_CACHE / meta["name"]
    return download_to(local_path, meta["id"], force=force)

# ★ FIX: moved to module level so both Power and Log tabs can access it
def find_subfolder_by_name(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
    """Find a subfolder by case-insensitive name."""
    kids = list_children(root_id, max_items=2000)
    for kid in kids:
        if kid.get("mimeType") == "application/vnd.google-apps.folder" and kid.get("name","").lower() == name_ci.lower():
            return kid
    return None



# ============================================================================
# Power log helpers (moved to module level)
# ============================================================================
LOG_RE = re.compile(r"^power_history_(\d{8})_(\d{6})\.log$", re.IGNORECASE)

@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def list_power_logs(folder_id: str, cache_epoch: str, nocache: str = "stable") -> List[Dict[str, Any]]:
    kids = list_children(folder_id, max_items=2000)
    files = [k for k in kids if k.get("mimeType") != "application/vnd.google-apps.folder" and LOG_RE.match(k.get("name",""))]
    files.sort(key=lambda m: m.get("name",""), reverse=True)
    return files

@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def ensure_log_cached(meta: Dict[str, Any], force: bool = False) -> Path:
    local_path = POWER_CACHE / meta["name"]
    return download_to(local_path, meta["id"], force=force)

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
    lines_raw = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines_raw: return None

    head_dt = None
    for l in lines_raw[:3]:
        m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", l)
        if m:
            try:
                head_dt = datetime.strptime(m.group(0), "%Y-%m-%d %H:%M:%S")
                break
            except Exception:
                pass
    if head_dt is None:
        return None

    wh_line   = next((l for l in lines_raw if l.upper().startswith("PH_WH")),   "")
    mah_line  = next((l for l in lines_raw if l.upper().startswith("PH_MAH")),  "")
    soci_line = next((l for l in lines_raw if l.upper().startswith("PH_SOCI")), "")
    socv_line = next((l for l in lines_raw if l.upper().startswith("PH_SOCV")), "")

    WH   = _parse_float_list(wh_line)
    mAh  = _parse_float_list(mah_line)
    SoCi = _parse_float_list(soci_line)
    SoCv = _parse_float_list(socv_line)

    L = max(len(WH), len(mAh), len(SoCi), len(SoCv))
    if L == 0: return None

    def _pad_left(arr, n):
        return [np.nan]*(n-len(arr)) + arr if len(arr) < n else arr[:n]

    WH, mAh, SoCi, SoCv = _pad_left(WH,L), _pad_left(mAh,L), _pad_left(SoCi,L), _pad_left(SoCv,L)
    times = [head_dt - timedelta(hours=(L-1 - i)) for i in range(L)]
    df = pd.DataFrame({"t": times, "PH_WH": WH, "PH_mAh": mAh, "PH_SoCi": SoCi, "PH_SoCv": SoCv})

    eps = 1e-9
    mask = (np.abs(df["PH_WH"])<eps) & (np.abs(df["PH_mAh"])<eps) & (np.abs(df["PH_SoCi"])<eps) & (np.abs(df["PH_SoCv"])<eps)
    return df[~mask].reset_index(drop=True)


# ============================================================================
# Helper: humanize chunk filenames
# ============================================================================
def _humanize_chunk(chunk_name: str) -> Dict[str, str]:
    """Parse a chunk filename into human-readable fields."""
    info = _parse_chunk_filename(chunk_name) if 'CHUNK_RE' in dir() else None
    m = CHUNK_RE.match(chunk_name or "")
    if not m:
        return {"species": chunk_name, "confidence": "—", "segment": "—", "date": "—", "source": "—"}
    try:
        root = m.group("root")
        tag = m.group("tag").upper()
        s = float(m.group("s"))
        e = float(m.group("e"))
        label = m.group("label").replace("_", " ")
        conf = float(m.group("conf"))
        # Parse date from root (YYYYMMDD_HHMMSS)
        try:
            dt = datetime.strptime(root, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = root
        return {
            "species": label,
            "confidence": f"{conf:.1%}",
            "segment": f"{s:.1f}s → {e:.1f}s",
            "date": date_str,
            "source": "KōreroNET" if tag == "KN" else "BirdNET",
        }
    except Exception:
        return {"species": chunk_name, "confidence": "—", "segment": "—", "date": "—", "source": "—"}


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("### 🐦 KōreroNET")
    st.caption("Bird Acoustic Monitoring")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🗺️ Nodes", "📊 Detections", "🎧 Verify", "⚡ Power", "📋 Log", "🔍 Search"],
        label_visibility="collapsed",
    )

    st.divider()
    node = st.selectbox("Active node", NODE_KEYS, index=0, key="node_select_top")
    if node != st.session_state.get("active_node"):
        st.session_state["active_node"] = node

    if st.button("↻ Refresh data", use_container_width=True, key=k("btn_refresh")):
        try:
            st.cache_data.clear()
            for _k in list(st.session_state.keys()):
                if str(_k).startswith("drive_kids::") or str(_k).startswith("DRIVE_EPOCH"):
                    del st.session_state[_k]
            st.session_state.pop("verify_loaded", None)
            st.rerun()
        except Exception as e:
            if type(e).__name__ in ("RerunException", "StopException", "RerunData"):
                raise
            st.warning(f"Refresh error: {e!s}")

    st.divider()

    # Quick summary in sidebar
    try:
        mode, chosen_date, df_summary = _latest_root_summary(GDRIVE_FOLDER_ID, False)
        if df_summary is not None and not df_summary.empty and chosen_date is not None:
            counts_sb = df_summary["Label"].astype(str).value_counts()
            top_sp = counts_sb.index[0] if len(counts_sb) > 0 else "—"
            st.caption(f"Latest: {_human_day(chosen_date)}")
            st.caption(f"Top species: **{top_sp}** ({int(counts_sb.iloc[0]):,})")
            st.caption(f"Total: {int(counts_sb.sum()):,} detections")
    except Exception:
        pass

    st.divider()
    st.caption("v2.0 · AUT · KōreroNET")


# ============================================================================
# Helper: load CSV index (shared by Detections + Search)
# ============================================================================
@st.cache_data(ttl=300, max_entries=5, show_spinner=False)
def _load_csv_index(_epoch: str):
    """Load and index all root CSVs. Returns (bn_paths, kn_paths, bn_by_date, kn_by_date)."""
    if drive_enabled():
        bn_meta, kn_meta = list_csvs_drive_root(GDRIVE_FOLDER_ID, cache_epoch=_epoch, nocache="stable")
        bn_paths = [str(ensure_csv_cached(m, subdir="root/bn", cache_epoch=_epoch, force=False)) for m in bn_meta]
        kn_paths = [str(ensure_csv_cached(m, subdir="root/kn", cache_epoch=_epoch, force=False)) for m in kn_meta]
    else:
        bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
        bn_paths, kn_paths = list(bn_local), list(kn_local)
    bn_by_date = build_date_index(bn_paths, "bn") if bn_paths else {}
    kn_by_date = build_date_index(kn_paths, "kn") if kn_paths else {}
    return bn_paths, kn_paths, bn_by_date, kn_by_date


# ============================================================================
# PAGE: Nodes
# ============================================================================
if page == "🗺️ Nodes":
    st.markdown("#### Node locations")

    _df_nodes = pd.DataFrame(
        [{"lat": n["lat"], "lon": n["lon"], "name": n["name"], "key": n["key"], "desc": n["desc"]} for n in NODES]
    )
    if "active_node" not in st.session_state:
        st.session_state["active_node"] = NODE_KEYS[0]

    try:
        _center = _df_nodes[_df_nodes["key"] == st.session_state["active_node"]][["lat", "lon"]].iloc[0].to_dict()
        center_lat, center_lon = float(_center["lat"]), float(_center["lon"])
    except Exception:
        center_lat, center_lon = -36.9003, 174.8839

    # Map + status side by side
    col_map, col_status = st.columns([3, 1])

    with col_map:
        rendered = False
        try:
            import folium
            from streamlit_folium import st_folium
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True, tiles="CartoDB Positron")
            for _, r in _df_nodes.iterrows():
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])], radius=12,
                    color="#2E7D32", fill=True, fill_color="#4CAF50", fill_opacity=0.8,
                    tooltip=f"<b>{r['name']}</b><br>{r['desc']}",
                ).add_to(m)
            st_folium(m, width=None, height=480, returned_objects=[])
            rendered = True
        except Exception:
            pass
        if not rendered:
            try:
                import pydeck as pdk
                layer = pdk.Layer("ScatterplotLayer", data=_df_nodes, get_position='[lon, lat]',
                                 get_radius=60, get_fill_color='[46,125,50,220]', pickable=True)
                view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                         tooltip={"text": "{name}\n{desc}"}), use_container_width=True)
                rendered = True
            except Exception:
                st.caption("⚠️ Interactive map unavailable — using fallback.")
        if not rendered:
            fig = go.Figure(go.Scattermapbox(
                lat=_df_nodes["lat"], lon=_df_nodes["lon"], mode="markers",
                marker=dict(size=14, color="#4CAF50"), text=_df_nodes["name"], hoverinfo="text",
            ))
            fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=12),
                              height=480, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_status:
        st.markdown("##### Node status")
        # Try to get latest power data for status
        try:
            if drive_enabled():
                logs_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
                if logs_folder:
                    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
                    p_files = list_power_logs(logs_folder["id"], cache_epoch=cache_epoch)
                    if p_files:
                        local = ensure_log_cached(p_files[0], force=False)
                        pdf = parse_power_log(local)
                        if pdf is not None and not pdf.empty:
                            last = pdf.iloc[-1]
                            st.metric("Status", "🟢 Online")
                            st.metric("Battery (SoC)", f"{last['PH_SoCi']:.0f}%")
                            st.metric("Energy", f"{last['PH_WH']:.1f} Wh")
                            st.metric("Last data", pdf["t"].max().strftime("%H:%M, %d %b"))
                        else:
                            st.metric("Status", "⚠️ No data")
                    else:
                        st.metric("Status", "⚠️ No logs")
                else:
                    st.metric("Status", "⚠️ No logs folder")
            else:
                st.metric("Status", "Offline mode")
        except Exception:
            st.metric("Status", "⚠️ Unknown")

    # Technology highlights
    with st.expander("About KōreroNET", expanded=False):
        features = [
            ('🔊', 'Bioacoustic monitoring of all vocal species in New Zealand wildlife.'),
            ('🎧', 'Full-spectrum recording: ultrasonic and audible ranges.'),
            ('🤖', 'Autonomous detection powered by our in-house AI models.'),
            ('🎛️', 'On-device edge computing and recording in a single package.'),
            ('📡', 'Deployable in remote areas with flexible connectivity.'),
            ('📶', 'Supports LoRaWAN, Wi-Fi and LTE networking.'),
            ('☀️', 'Solar-powered and weather-sealed for harsh environments.'),
            ('⚡', 'Energy-efficient: records and processes in intervals to save power.'),
            ('🐦', 'Detects both pests and birds of interest.'),
            ('📁', 'Provides accessible recordings of species of interest.')
        ]
        feat_html = '<div class="features-grid">'
        for icon, text in features:
            feat_html += f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-text">{text}</div></div>'
        feat_html += '</div>'
        st.markdown(feat_html, unsafe_allow_html=True)


# ============================================================================
# PAGE: Detections (reactive — no Update button)
# ============================================================================
elif page == "📊 Detections":
    st.markdown("#### Species detections")
    st.caption("Heatmap of detections by species and hour. Filters auto-apply.")

    epoch = st.session_state.get("DRIVE_EPOCH", "0")
    bn_paths, kn_paths, bn_by_date, kn_by_date = _load_csv_index(epoch)

    bn_dates = sorted(bn_by_date.keys())
    kn_dates = sorted(kn_by_date.keys())
    paired_dates = sorted(set(bn_dates) & set(kn_dates))

    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        src = st.selectbox("Source", ["KōreroNET (kn)", "BirdNET (bn)", "Combined"], index=0, key=k("det_src"))
    with fc3:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01, key=k("det_conf"))

    if src == "Combined":
        options = paired_dates
    elif src == "BirdNET (bn)":
        options = bn_dates
    else:
        options = kn_dates

    if not options:
        st.warning(f"No available dates for {src}.")
    else:
        with fc2:
            d = st.date_input("Day", value=options[-1], min_value=options[0], max_value=options[-1], key=k("det_day"))

        if d not in set(options):
            earlier = [x for x in options if x <= d]
            d = earlier[-1] if earlier else options[0]
            st.caption(f"Snapped to nearest: **{d.isoformat()}**")

        def _load_filter(kind, day):
            idx = bn_by_date if kind == "bn" else kn_by_date
            frames = []
            for p in idx.get(day, []):
                try:
                    raw = load_csv(p)
                    std = standardize_root_df(raw, kind)
                    std = std[std["ActualTime"].dt.date == day]
                    if not std.empty: frames.append(std)
                except Exception:
                    pass
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if src == "BirdNET (bn)":
            df = _load_filter("bn", d)
            title = f"BirdNET — {d.isoformat()}"
        elif src == "KōreroNET (kn)":
            df = _load_filter("kn", d)
            title = f"KōreroNET — {d.isoformat()}"
        else:
            df = pd.concat([_load_filter("bn", d), _load_filter("kn", d)], ignore_index=True)
            title = f"Combined — {d.isoformat()}"

        df_f = df[pd.to_numeric(df.get("Confidence", 0), errors="coerce") >= min_conf].copy() if not df.empty else pd.DataFrame()

        if df_f.empty:
            st.info("No detections above the confidence threshold for this date.")
        else:
            df_f["Hour"] = df_f["ActualTime"].dt.hour
            hour_labels = {h: f"{(h%12) or 12} {'AM' if h<12 else 'PM'}" for h in range(24)}
            order = [hour_labels[h] for h in range(24)]
            df_f["HL"] = df_f["Hour"].map(hour_labels)
            pivot = df_f.groupby(["Label","HL"]).size().unstack(fill_value=0).astype(int)
            for lbl in order:
                if lbl not in pivot.columns: pivot[lbl] = 0
            pivot = pivot[order]
            totals = pivot.sum(axis=1)
            pivot = pivot.loc[totals.sort_values(ascending=False).index]

            fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                            color_continuous_scale="YlOrRd",
                            labels=dict(x="Hour", y="Species", color="Count"),
                            text_auto=True, aspect="auto")
            fig.update_layout(title_text=title, margin=dict(l=10,r=10,t=45,b=10),
                              coloraxis_colorbar=dict(thickness=14, len=0.6))
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="kn-metrics">
                <div class="kn-metric"><div class="kn-metric-val">{int(totals.sum()):,}</div><div class="kn-metric-label">Total detections</div></div>
                <div class="kn-metric"><div class="kn-metric-val">{len(totals)}</div><div class="kn-metric-label">Species</div></div>
                <div class="kn-metric"><div class="kn-metric-val">{min_conf:.0%}</div><div class="kn-metric-label">Confidence ≥</div></div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# PAGE: Verify (humanized chunks, progress bar)
# ============================================================================
elif page == "🎧 Verify":
    st.markdown("#### Verify recordings")
    st.caption("Pick a date, then browse and play audio chunks for that snapshot.")

    if not drive_enabled():
        st.error("Google Drive is not configured.")
    else:
        NAV_COOLDOWN_SECONDS = 0.4
        if "tab2_last_play_ts" not in st.session_state:
            st.session_state["tab2_last_play_ts"] = 0.0

        backup = _find_backup_folder(GDRIVE_FOLDER_ID)
        if not backup:
            st.warning("No `Backup` folder found under Drive root.")
        else:
            snap_folders = [kid for kid in list_children(backup["id"], max_items=2000)
                            if kid.get("mimeType") == "application/vnd.google-apps.folder"
                            and SNAP_RE.match(kid.get("name", ""))]
            snap_folders.sort(key=lambda m: m.get("name", ""), reverse=True)

            if not snap_folders:
                st.warning("No snapshot folders found.")
            else:
                snap_date_map = {}
                for sf in snap_folders:
                    sd = _parse_date_from_snapname(sf["name"])
                    if sd: snap_date_map[sd] = sf
                avail_dates = sorted(snap_date_map.keys(), reverse=True)

                if not avail_dates:
                    st.info("No valid snapshot dates found.")
                else:
                    fc1, fc2, fc3 = st.columns([2, 2, 1])
                    with fc1:
                        day_pick = st.date_input("Snapshot date", value=avail_dates[0],
                                                  min_value=avail_dates[-1], max_value=avail_dates[0], key=k("v_day"))
                    with fc2:
                        src_mode_v = st.selectbox("Source", ["KoreroNET (KN)", "BirdNET (BN)", "Combined"], key=k("v_src"))
                    with fc3:
                        min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01, key=k("v_conf"))

                    if day_pick not in snap_date_map:
                        nearest = min(avail_dates, key=lambda d: abs((d - day_pick).days))
                        day_pick = nearest
                        st.caption(f"Snapped to nearest: **{day_pick.isoformat()}**")

                    sn = snap_date_map[day_pick]
                    snap_id, snap_name = sn["id"], sn["name"]

                    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
                    snap_kids = list_children(snap_id, max_items=2000)
                    chunk_dirs = _find_chunk_dirs(snap_id, kids=snap_kids)

                    rows_frames = []
                    if day_pick >= CUTOFF_NEW:
                        for kind_key, csv_name in [("KN", "koreronet_master.csv"), ("BN", "birdnet_master.csv")]:
                            if src_mode_v == "KoreroNET (KN)" and kind_key == "BN": continue
                            if src_mode_v == "BirdNET (BN)" and kind_key == "KN": continue
                            meta = _find_named_in_snapshot(snap_id, csv_name, kids=snap_kids)
                            if not meta: continue
                            csv_path = ensure_csv_cached(meta, subdir=f"snap_{snap_id}/{kind_key.lower()}", cache_epoch=cache_epoch)
                            try:
                                df = pd.read_csv(csv_path)
                                lc = "Label" if "Label" in df.columns else ("Common name" if "Common name" in df.columns else None)
                                cc = "Probability" if "Probability" in df.columns else ("Confidence" if "Confidence" in df.columns else None)
                                frame = pd.DataFrame({
                                    "Kind": kind_key,
                                    "Label": df[lc].astype(str) if lc else "Unknown",
                                    "Confidence": pd.to_numeric(df[cc], errors="coerce") if cc else np.nan,
                                    "Start": np.nan, "End": np.nan, "WavBase": "",
                                    "ChunkName": df.get("Clip", "").astype(str).str.strip(),
                                    "ChunkDriveFolderId": chunk_dirs[kind_key],
                                })
                                rows_frames.append(frame)
                            except Exception:
                                pass
                    else:
                        for kind_key in (["KN","BN"] if src_mode_v=="Combined" else ["KN"] if src_mode_v=="KoreroNET (KN)" else ["BN"]):
                            files_only = [f for f in snap_kids if f.get("mimeType") != "application/vnd.google-apps.folder"]
                            legacy = [f for f in files_only if _match_legacy_master_name(f.get("name",""), kind_key)]
                            if not legacy:
                                for sf in [f for f in snap_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]:
                                    cand = [f for f in list_children(sf["id"], max_items=2000)
                                            if f.get("mimeType") != "application/vnd.google-apps.folder"
                                            and _match_legacy_master_name(f.get("name",""), kind_key)]
                                    if cand: legacy = cand; break
                            if not legacy: continue
                            legacy.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
                            csv_path = ensure_csv_cached(legacy[0], subdir=f"snap_{snap_id}/{kind_key.lower()}", cache_epoch=cache_epoch)
                            try:
                                df = pd.read_csv(csv_path)
                                wav_col = df.get("File", pd.Series(dtype=str)).astype(str).apply(os.path.basename)
                                if kind_key == "BN":
                                    s_col = pd.to_numeric(df.get("Start (s)", df.get("Start", np.nan)), errors="coerce")
                                    e_col = pd.to_numeric(df.get("End (s)", df.get("End", np.nan)), errors="coerce")
                                    lab_col = df.get("Common name", df.get("Label", "Unknown")).astype(str)
                                else:
                                    s_col = pd.to_numeric(df.get("Start", df.get("Start (s)", np.nan)), errors="coerce")
                                    e_col = pd.to_numeric(df.get("End", df.get("End (s)", np.nan)), errors="coerce")
                                    lab_col = df.get("Label", "Unknown").astype(str)
                                conf_col = pd.to_numeric(df.get("Confidence", np.nan), errors="coerce")
                                chunk_names = [_compose_chunk_name(kind_key.lower(), w, s, e, l, c)
                                               for w, s, e, l, c in zip(wav_col, s_col, e_col, lab_col, conf_col)]
                                frame = pd.DataFrame({
                                    "Kind": kind_key, "Label": lab_col, "Confidence": conf_col,
                                    "Start": s_col, "End": e_col, "WavBase": wav_col,
                                    "ChunkName": chunk_names, "ChunkDriveFolderId": chunk_dirs[kind_key],
                                })
                                rows_frames.append(frame)
                            except Exception:
                                pass

                    if not rows_frames:
                        st.info(f"No data for {day_pick.isoformat()} with source {src_mode_v}.")
                    else:
                        master = pd.concat(rows_frames, ignore_index=True)
                        master = master[pd.to_numeric(master["Confidence"], errors="coerce") >= float(min_conf_v)]

                        if master.empty:
                            st.info("No detections above the selected confidence.")
                        else:
                            counts = master.groupby("Label").size().sort_values(ascending=False)
                            species_list = list(counts.index)

                            species = st.selectbox("Species", species_list,
                                                    format_func=lambda s: f"{s} — {counts[s]:,} detections", key=k("v_sp"))

                            playlist = master[master["Label"] == species].sort_values(["Kind","ChunkName"]).reset_index(drop=True)

                            if playlist.empty:
                                st.info("No clips for this species.")
                            else:
                                idx_key = f"v_idx::{day_pick}::{src_mode_v}::{species}"
                                if idx_key not in st.session_state:
                                    st.session_state[idx_key] = 0
                                idx = st.session_state[idx_key] % len(playlist)

                                # Progress bar
                                st.progress((idx + 1) / len(playlist))
                                st.caption(f"Detection {idx + 1} of {len(playlist)}")

                                col1, col2, col3 = st.columns([1, 1, 1])
                                do_play = False
                                with col1:
                                    if st.button("⏮ Prev", key=k("v_prev"), use_container_width=True):
                                        idx = (idx - 1) % len(playlist)
                                with col2:
                                    if st.button("▶ Play", key=k("v_play"), type="primary", use_container_width=True):
                                        now_ts = time.time()
                                        if now_ts - st.session_state["tab2_last_play_ts"] >= NAV_COOLDOWN_SECONDS:
                                            st.session_state["tab2_last_play_ts"] = now_ts
                                            do_play = True
                                        else:
                                            st.info("Just a moment between plays.")
                                with col3:
                                    if st.button("⏭ Next", key=k("v_next"), use_container_width=True):
                                        idx = (idx + 1) % len(playlist)

                                st.session_state[idx_key] = idx
                                row = playlist.iloc[idx]

                                # Humanized chunk display
                                chunk_info = _humanize_chunk(str(row.get("ChunkName", "")))
                                conf_val = float(row["Confidence"]) if pd.notna(row["Confidence"]) else 0

                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown(f"**Species:** {chunk_info['species']}")
                                    st.markdown(f"**Confidence:** {conf_val:.1%}")
                                with c2:
                                    st.markdown(f"**Segment:** {chunk_info['segment']}")
                                    st.markdown(f"**Date:** {day_pick} · **Source:** {row['Kind']}")

                                @_retry()
                                def _safe_chunk(cn, fid, sd):
                                    return ensure_chunk_cached(cn, fid, subdir=sd, force=False)

                                if do_play:
                                    try:
                                        cn = str(row.get("ChunkName", "") or "")
                                        fid = str(row.get("ChunkDriveFolderId", "") or "")
                                        if not cn or not fid:
                                            st.warning("No chunk mapping available.")
                                        else:
                                            with st.spinner("Fetching audio…"):
                                                cached = _safe_chunk(cn, fid, row["Kind"])
                                            if cached and cached.exists() and cached.stat().st_size > 0:
                                                if cached.stat().st_size > 30*1024*1024:
                                                    st.warning("Audio file too large (>30 MB).")
                                                else:
                                                    try:
                                                        st.audio(cached.read_bytes(), format="audio/wav")
                                                    except MemoryError:
                                                        st.warning("Not enough memory for preview.")
                                            else:
                                                st.warning("Audio chunk not found.")
                                    except Exception as e:
                                        if type(e).__name__ in ("RerunException","StopException","RerunData"): raise
                                        st.warning(f"Playback error: {e!s}")


# ============================================================================
# PAGE: Power (auto-load)
# ============================================================================
elif page == "⚡ Power":
    st.markdown("#### Node power history")
    st.caption("State of Charge and energy over the last 7 days.")

    if not drive_enabled():
        st.error("Google Drive is not configured.")
    else:
        logs_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
        if not logs_folder:
            st.warning("No `Power logs` folder found.")
        else:
            # Auto-load — no button required
            with st.spinner("Loading power data…"):
                cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
                files = list_power_logs(logs_folder["id"], cache_epoch=cache_epoch)
                cutoff = datetime.now() - timedelta(days=7)
                frames = []
                for meta in files:
                    local = ensure_log_cached(meta, force=False)
                    df = parse_power_log(local)
                    if df is None or df.empty: continue
                    df = df[df["t"] >= cutoff]
                    if df.empty: break
                    frames.append(df)
                ts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["t","PH_WH","PH_mAh","PH_SoCi","PH_SoCv"])

            if ts.empty:
                st.info("No parsable power logs in the last 7 days.")
            else:
                ts = ts.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_SoCi"], mode="lines",
                                         name="SoC (%)", line=dict(color="#4CAF50", width=2.5)))
                fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_WH"], mode="lines",
                                         name="Energy (Wh)", line=dict(color="#FF9800", width=2), yaxis="y2"))
                fig.update_layout(
                    yaxis2=dict(overlaying="y", side="right"),
                    legend=dict(orientation="h", y=-0.12),
                    margin=dict(l=10,r=10,t=10,b=10),
                    hovermode="x unified",
                )
                fig.update_layout(title_text="Battery & Energy — last 7 days")
                fig.update_yaxes(title_text="SoC (%)", range=[0,105], selector=dict(side="left"))
                fig.update_yaxes(title_text="Wh", selector=dict(side="right"))
                st.plotly_chart(fig, use_container_width=True)

                last = ts.iloc[-1]
                st.markdown(f"""
                <div class="kn-metrics">
                    <div class="kn-metric"><div class="kn-metric-val">{last['PH_SoCi']:.1f}%</div><div class="kn-metric-label">Current SoC</div></div>
                    <div class="kn-metric"><div class="kn-metric-val">{last['PH_WH']:.1f} Wh</div><div class="kn-metric-label">Current energy</div></div>
                    <div class="kn-metric"><div class="kn-metric-val">{len(ts):,}</div><div class="kn-metric-label">Data points</div></div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Range: {ts['t'].min().strftime('%Y-%m-%d %H:%M')} → {ts['t'].max().strftime('%Y-%m-%d %H:%M')}")


# ============================================================================
# PAGE: Log (structured viewer)
# ============================================================================
elif page == "📋 Log":
    st.markdown("#### GUI autostart log")
    st.caption("Latest autostart log from the node, parsed into a structured view.")

    if not drive_enabled():
        st.error("Google Drive is not configured.")
    else:
        power_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
        raw_folder = find_subfolder_by_name(power_folder["id"], "raw") if power_folder else None

        if not raw_folder:
            st.warning("Could not locate `Power logs/raw` in Drive.")
        else:
            cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
            files = list_autostart_logs(raw_folder["id"], cache_epoch=cache_epoch)

            if not files:
                st.info("No `*__gui_autostart.log` files found.")
            else:
                latest = files[0]
                local = ensure_raw_cached(latest, force=True)
                st.caption(f"File: `{latest.get('name', '?')}` — {local.stat().st_size:,} bytes")

                # Read and parse log
                try:
                    raw_text = local.read_text(encoding="utf-8", errors="replace")
                    raw_lines = raw_text.splitlines()[-500:]  # tail 500

                    # Try to parse structured log entries
                    log_rows = []
                    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*[-|]\s*(\w+)\s*[-|:]\s*(.*)")
                    for line in raw_lines:
                        m = ts_re.match(line.strip())
                        if m:
                            log_rows.append({"Timestamp": m.group(1), "Level": m.group(2), "Message": m.group(3)})
                        elif line.strip():
                            log_rows.append({"Timestamp": "", "Level": "—", "Message": line.strip()})

                    if log_rows:
                        log_df = pd.DataFrame(log_rows)

                        # Filter controls
                        fc1, fc2 = st.columns([1, 3])
                        with fc1:
                            levels = sorted(log_df["Level"].unique())
                            sel_levels = st.multiselect("Severity", levels, default=levels, key=k("log_levels"))
                        with fc2:
                            search_text = st.text_input("Search", placeholder="Filter messages…", key=k("log_search"))

                        filtered = log_df[log_df["Level"].isin(sel_levels)]
                        if search_text:
                            filtered = filtered[filtered["Message"].str.contains(search_text, case=False, na=False)]

                        st.dataframe(filtered.iloc[::-1], use_container_width=True, hide_index=True, height=500)
                    else:
                        # Fallback: raw code view
                        st.code("\n".join(reversed(raw_lines)), language="log")

                except Exception as e:
                    st.warning(f"Could not parse log: {e!s}")
                    st.code("(raw log unavailable)", language="log")

                st.download_button("Download full log", local.read_bytes(),
                                    file_name=latest.get("name", "log.txt"), mime="text/plain", key=k("dl_log"))


# ============================================================================
# PAGE: Search (species trends)
# ============================================================================
elif page == "🔍 Search":
    st.markdown("#### Species search & trends")
    st.caption("Search for species across a date range. See daily detection counts and peak activity hours.")

    epoch = st.session_state.get("DRIVE_EPOCH", "0")
    bn_paths, kn_paths, bn_by_date, kn_by_date = _load_csv_index(epoch)
    all_dates = sorted(set(bn_by_date.keys()) | set(kn_by_date.keys()))

    if not all_dates:
        st.warning("No detection data available.")
    else:
        # Species hint from latest date
        @st.cache_data(ttl=600, max_entries=5, show_spinner=False)
        def _species_hint(_bn, _kn, _d):
            sp = set()
            for kind, idx in [("bn", bn_by_date), ("kn", kn_by_date)]:
                for p in idx.get(_d, []):
                    try:
                        std = standardize_root_df(load_csv(p), kind)
                        sp.update(std["Label"].dropna().unique())
                    except Exception: pass
            return sorted(sp)

        known = _species_hint(tuple(bn_paths), tuple(kn_paths), all_dates[-1])

        st.markdown("**Enter species names** (comma-separated)")
        species_input = st.text_input("Species", placeholder="e.g. Tui, Morepork, Silvereye",
                                       key=k("s_sp"), label_visibility="collapsed")
        if known:
            st.caption(f"Available: {', '.join(known[:15])}{'…' if len(known)>15 else ''}")

        sc1, sc2, sc3, sc4 = st.columns([2, 2, 1, 1])
        with sc1:
            default_start = max(all_dates[0], all_dates[-1] - timedelta(days=30))
            d_start = st.date_input("Start", value=default_start, min_value=all_dates[0], max_value=all_dates[-1], key=k("s_s"))
        with sc2:
            d_end = st.date_input("End", value=all_dates[-1], min_value=all_dates[0], max_value=all_dates[-1], key=k("s_e"))
        with sc3:
            search_conf = st.slider("Confidence", 0.0, 1.0, 0.80, 0.01, key=k("s_c"))
        with sc4:
            search_src = st.selectbox("Source", ["Combined", "KōreroNET", "BirdNET"], key=k("s_src"))

        if st.button("Search", type="primary", use_container_width=True, key=k("s_go")):
            target_species = [s.strip() for s in species_input.split(",") if s.strip()]
            if not target_species:
                st.warning("Please enter at least one species name.")
            else:
                if search_src == "BirdNET":
                    date_indices = [("bn", bn_by_date)]
                elif search_src == "KōreroNET":
                    date_indices = [("kn", kn_by_date)]
                else:
                    date_indices = [("bn", bn_by_date), ("kn", kn_by_date)]

                dates_in_range = [d for d in all_dates if d_start <= d <= d_end]
                if not dates_in_range:
                    st.warning(f"No data between {d_start} and {d_end}.")
                else:
                    with st.spinner(f"Searching {len(dates_in_range)} dates…"):
                        paths_to_load = {}
                        for kind, idx in date_indices:
                            for d in dates_in_range:
                                for p in idx.get(d, []):
                                    if p not in paths_to_load: paths_to_load[p] = kind

                        frames = []
                        for p, kind in paths_to_load.items():
                            try:
                                raw = load_csv(p)
                                std = standardize_root_df(raw, kind)
                                std = std[(std["ActualTime"].dt.date >= d_start) & (std["ActualTime"].dt.date <= d_end)]
                                std = std[pd.to_numeric(std["Confidence"], errors="coerce") >= search_conf]
                                if not std.empty: frames.append(std)
                            except Exception: pass

                        if not frames:
                            st.info("No detections found.")
                        else:
                            df_all = pd.concat(frames, ignore_index=True)
                            matched = set()
                            for t in target_species:
                                matches = df_all[df_all["Label"].str.lower().str.contains(t.lower(), na=False)]["Label"].unique()
                                matched.update(matches)

                            if not matched:
                                st.info(f"No species matching **{', '.join(target_species)}**. Try partial names.")
                            else:
                                df = df_all[df_all["Label"].isin(matched)].copy()
                                df["Date"] = df["ActualTime"].dt.date
                                df["Hour"] = df["ActualTime"].dt.hour

                                st.markdown(f"""
                                <div class="kn-metrics">
                                    <div class="kn-metric"><div class="kn-metric-val">{len(df):,}</div><div class="kn-metric-label">Total detections</div></div>
                                    <div class="kn-metric"><div class="kn-metric-val">{len(matched)}</div><div class="kn-metric-label">Species matched</div></div>
                                    <div class="kn-metric"><div class="kn-metric-val">{df['Date'].nunique()}</div><div class="kn-metric-label">Days with data</div></div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Daily trend
                                st.markdown("##### Daily detections")
                                daily = df.groupby(["Date","Label"]).size().reset_index(name="Detections")
                                daily["Date"] = pd.to_datetime(daily["Date"])
                                full_dates = pd.date_range(d_start, d_end, freq="D")
                                filled = []
                                for sp in sorted(matched):
                                    sp_d = daily[daily["Label"]==sp].set_index("Date").reindex(full_dates, fill_value=0)
                                    sp_d["Label"] = sp
                                    sp_d.index.name = "Date"
                                    filled.append(sp_d.reset_index())
                                daily_f = pd.concat(filled, ignore_index=True)

                                fig = px.line(daily_f, x="Date", y="Detections", color="Label",
                                              markers=len(dates_in_range) <= 60)
                                fig.update_layout(legend=dict(orientation="h", y=-0.15),
                                                  margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified")
                                fig.update_traces(line=dict(width=2.5))
                                st.plotly_chart(fig, use_container_width=True)

                                # Hour heatmap
                                st.markdown("##### Peak activity by hour")
                                hourly = df.groupby(["Label","Hour"]).size().reset_index(name="Count")
                                hour_labels = {h: f"{(h%12) or 12} {'AM' if h<12 else 'PM'}" for h in range(24)}
                                pivot = hourly.pivot_table(index="Label", columns="Hour", values="Count", fill_value=0).astype(int)
                                for h in range(24):
                                    if h not in pivot.columns: pivot[h] = 0
                                pivot = pivot[sorted(pivot.columns)]
                                pivot.columns = [hour_labels[h] for h in pivot.columns]
                                totals = pivot.sum(axis=1)
                                pivot = pivot.loc[totals.sort_values(ascending=False).index]

                                fig2 = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                                                 color_continuous_scale="YlOrRd",
                                                 labels=dict(x="Hour", y="Species", color="Detections"),
                                                 text_auto=True, aspect="auto")
                                fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                                                   coloraxis_colorbar=dict(thickness=14, len=0.6))
                                fig2.update_xaxes(type="category")
                                st.plotly_chart(fig2, use_container_width=True)

                                # Peak hour table
                                st.markdown("##### Peak hour per species")
                                peak_data = []
                                for sp in sorted(matched):
                                    sp_h = df[df["Label"]==sp].groupby("Hour").size()
                                    if sp_h.empty: continue
                                    ph = sp_h.idxmax()
                                    total = sp_h.sum()
                                    days = df[df["Label"]==sp]["Date"].nunique()
                                    peak_data.append({
                                        "Species": sp, "Peak hour": hour_labels[ph],
                                        "At peak": int(sp_h.max()), "Total": int(total),
                                        "Days active": days, "Avg/day": f"{total/max(days,1):.1f}",
                                    })
                                if peak_data:
                                    st.dataframe(pd.DataFrame(peak_data), use_container_width=True, hide_index=True)
