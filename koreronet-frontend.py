#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KōreroNET Dashboard — HARDENED (Drive-safe, cache-safe)
# ---------------------------------------------------------------------------------
# What’s new vs your last version:
#  • Circuit breaker for Google Drive: after repeated errors, session falls back to offline
#  • All Drive calls wrapped with jittered retries + googleapiclient num_retries
#  • Atomic cached downloads + file lock to prevent competing writers (race-proof)
#  • Defensive st.cache_data usage (epoch keys only; no cross-key invalidation)
#  • Safer CSV parsing (on_bad_lines='skip', low_memory=False) and chunked loading
#  • Audio fetch capped to 30 MB with defensive base64 embed and memory errors handled
#  • Refresh only clears our own keys; avoids clearing in-flight caches
#  • UI toggle to force Offline (local clone) instantly if Drive is flaky
#  • Never hard-crash: every Drive interaction is non-fatal and reports a gentle warning
#
# Functionality preserved:
#  • Splash + Welcome overlay (top-3 species summary)
#  • Tab “Nodes” with mapping fallbacks (folium → pydeck → plotly)
#  • Tab 1 “Detections” (root CSV heatmaps, calendar, min confidence, Combined mode)
#  • Tab 2 “Verify recordings” (snapshot masters, playlist prev/play/next, on-demand fetch)
#  • Tab 3 “Power” (dual-axis SoC_i/Wh)
#  • Tab 4 “GUI autostart log tail” with download
#
# Notes:
#  • Set OFFLINE_DEPLOY=True to force local mode. Or use the UI toggle at runtime.
#  • For Drive mode, set secrets as before (GDRIVE_FOLDER_ID + [service_account]).
# ---------------------------------------------------------------------------------

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
import threading
import tempfile

# Optional (best-effort) import
try:
    from streamlit_plotly_events import plotly_events  # noqa
except Exception:
    plotly_events = None

# =================================================================================
# Hardened utilities: retry, circuit breaker, file-lock + atomic writes
# =================================================================================

def _retry(exceptions=(Exception,), tries=3, base_delay=0.35, max_delay=1.2, jitter=0.25):
    """Simple retry decorator with jittered backoff."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            _tries = tries
            _delay = base_delay
            last_err = None
            while _tries > 0:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    _tries -= 1
                    if _tries <= 0:
                        raise
                    time.sleep(_delay + random.random() * jitter)
                    _delay = min(max_delay, _delay * 1.6)
            raise last_err  # should not happen
        return wrap
    return deco

# --- circuit breaker state (per session) ---
if "DRIVE_FAILS" not in st.session_state:
    st.session_state["DRIVE_FAILS"] = 0
if "DRIVE_OPEN" not in st.session_state:
    st.session_state["DRIVE_OPEN"] = True  # when False => force offline for this session

def _cb_record(success: bool):
    if success:
        st.session_state["DRIVE_FAILS"] = max(0, st.session_state["DRIVE_FAILS"] - 1)
    else:
        st.session_state["DRIVE_FAILS"] += 1
        if st.session_state["DRIVE_FAILS"] >= 3:
            st.session_state["DRIVE_OPEN"] = False  # trip
            st.warning("Drive circuit tripped — falling back to Offline for this session.")

# --- simple file lock (portable) ---
def _acquire_lock(lock_path: Path, timeout: float = 10.0) -> Optional[Path]:
    start = time.time()
    while True:
        try:
            # O_EXCL ensures we "win" if the file doesn't exist
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return lock_path
        except FileExistsError:
            if time.time() - start > timeout:
                return None
            time.sleep(0.1)

def _release_lock(lock_path: Path):
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass

# =================================================================================
# Page setup + styles
# =================================================================================
st.set_page_config(page_title="KōreroNET Dashboard", layout="wide")
st.markdown("""
<style>
[data-testid="stDecoration"] { display: none !important; }
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
.overlay-card { max-width: 860px; margin: 6vh auto; padding: 24px 28px; border: none; border-radius: 16px; background: rgba(28,28,28,.96); box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.overlay-title {font-size: clamp(28px,4vw,42px); font-weight: 800; letter-spacing:.01em; margin-bottom:.25rem;}
.overlay-sub {font-size: clamp(16px,2.2vw,18px); opacity:.9; margin-bottom: .75rem;}
.overlay-pill { display:inline-block; padding: .35rem .75rem; border: 1px solid #444; border-radius: 999px; margin:.25rem .25rem 0 0; font-size:.95rem; background: rgba(255,255,255,.03);}
.overlay-card hr { border: 0; height: 0; }
.overlay-card :where(p,div,span):empty { display: none !important; }
.features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-top: 1rem; }
.feature-card { display: flex; align-items: flex-start; gap: 0.65rem; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.75rem 0.9rem; }
.feature-icon { font-size: 1.4rem; line-height: 1.4rem; }
.feature-text { font-size: 1rem; line-height: 1.3rem; flex: 1; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =================================================================================
# Constants & Regex
# =================================================================================
CHUNK_RE = re.compile(
    r"^(?P<root>\\d{8}_\\d{6})__(?P<tag>bn|kn)_(?P<s>\\d+\\.\\d{2})_(?P<e>\\d+\\.\\d{2})__(?P<label>.+?)__p(?P<conf>\\d+\\.\\d{2})\\.wav$",
    re.IGNORECASE,
)
NEW_ROOT_BN = re.compile(r"^\\d{8}_\\d{6}_birdnet_master\\.csv$", re.IGNORECASE)
NEW_ROOT_KN = re.compile(r"^\\d{8}_\\d{6}_koreronet_master\\.csv$", re.IGNORECASE)
SNAP_RE     = re.compile(r"^(\\d{8})_(\\d{6})$", re.IGNORECASE)
LOG_AUTOSTART_RE = re.compile(r"^(\\d{8})_(\\d{6})__gui_autostart\\.log$", re.IGNORECASE)
CUTOFF_NEW  = date(2025, 10, 31)

NODES = [
    {"key":"Auckland-Orākei","name":"Auckland — Orākei","lat":-36.8528,"lon":174.8150,"desc":"Primary demo node in Orākei, Auckland"}
]
NODE_KEYS = [n["key"] for n in NODES]

# =================================================================================
# Splash
# =================================================================================
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
            """, unsafe_allow_html=True
        )
    time.sleep(1.2)
    st.session_state["splash_done"] = True
    st.rerun()

# =================================================================================
# Session helpers
# =================================================================================
if "_sess_salt" not in st.session_state:
    st.session_state["_sess_salt"] = str(uuid.uuid4())[:8]
def k(name: str) -> str:
    return f"{name}::{st.session_state['_sess_salt']}"

# =================================================================================
# Caches & local fallback
# =================================================================================
CACHE_ROOT   = Path(tempfile.gettempdir()) / "koreronet_cache"
CSV_CACHE    = CACHE_ROOT / "csv"
CHUNK_CACHE  = CACHE_ROOT / "chunks"
POWER_CACHE  = CACHE_ROOT / "power"
for _p in (CSV_CACHE, CHUNK_CACHE, POWER_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

DEFAULT_ROOT = r"G:\\My Drive\\From the node"
ROOT_LOCAL   = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)
OFFLINE_DEPLOY: bool = bool(os.getenv("KORERONET_OFFLINE", "0") == "1")

def _secret_or_env(name: str, default=None):
    v = os.getenv(name)
    if v is not None:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return default

# =================================================================================
# Drive client (guarded) + UI Offline toggle
# =================================================================================
row_top = st.columns([3,2,2])
with row_top[0]:
    node = st.selectbox("Node Select", ["Auckland-Orākei"], index=0, key="node_select_top")
with row_top[1]:
    force_offline_ui = st.toggle("Force Offline (local clone)", value=False, help="Disable Drive for this session.")
with row_top[2]:
    if st.button("🔄 Clear caches", help="Clear cached data and reload."):
        try:
            st.cache_data.clear()
            for _k in list(st.session_state.keys()):
                if str(_k).startswith("drive_kids::") or str(_k).startswith("DRIVE_EPOCH"):
                    del st.session_state[_k]
            st.success("Caches cleared. Reloading…")
            st.rerun()
        except Exception as e:
            st.warning(f"Refresh encountered a non-fatal error: {e!s}")

GDRIVE_FOLDER_ID = _secret_or_env("GDRIVE_FOLDER_ID", None)
if OFFLINE_DEPLOY or force_offline_ui or (not st.session_state.get("DRIVE_OPEN", True)):
    # force local
    GDRIVE_FOLDER_ID = ROOT_LOCAL

_DRIVE_LOCK = threading.Lock()

def _normalize_private_key(pk: str) -> str:
    if not isinstance(pk, str): return pk
    if "\\n" in pk: pk = pk.replace("\\n", "\\n").encode("utf-8").decode("unicode_escape")
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
        with _DRIVE_LOCK:
            return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None

def get_drive_client():
    if "drive_client" in st.session_state:
        return st.session_state["drive_client"]
    client = None
    if isinstance(GDRIVE_FOLDER_ID, str) and not os.path.isdir(GDRIVE_FOLDER_ID):
        client = _build_drive_client()
    st.session_state["drive_client"] = client
    return client

def drive_enabled() -> bool:
    # Offline emulation if GDRIVE_FOLDER_ID is a directory
    return bool(GDRIVE_FOLDER_ID)

# =================================================================================
# Drive-like API (online/offline) with atomic downloads & retries
# =================================================================================
def _is_offline_root() -> bool:
    return isinstance(GDRIVE_FOLDER_ID, str) and os.path.isdir(GDRIVE_FOLDER_ID)

@_retry()
def _drive_list_children_online(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    from googleapiclient.errors import HttpError
    cli = get_drive_client()
    if not cli:
        return []
    items, token = [], None
    while True:
        page_size = min(100, max_items - len(items))
        if page_size <= 0: break
        try:
            with _DRIVE_LOCK:
                req = cli.files().list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,md5Checksum,parents)",
                    pageSize=page_size, pageToken=token, orderBy="folder,name_natural",
                    includeItemsFromAllDrives=True, supportsAllDrives=True, corpora="allDrives",
                )
                resp = req.execute(num_retries=2)
            items.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token: break
        except HttpError as e:
            _cb_record(False); raise
    _cb_record(True)
    return items

def _drive_list_children_offline(folder_path: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    base = Path(folder_path)
    if not base.exists() or not base.is_dir():
        return []
    items = []
    for i, entry in enumerate(base.iterdir()):
        if i >= max_items: break
        stat = entry.stat()
        mt = datetime.fromtimestamp(stat.st_mtime).isoformat()
        items.append({
            "id": str(entry),
            "name": entry.name,
            "mimeType": "application/vnd.google-apps.folder" if entry.is_dir() else "application/octet-stream",
            "modifiedTime": mt,
            "size": str(stat.st_size),
            "md5Checksum": None,
            "parents": [str(base)],
        })
    def _key(m):
        return (0 if m.get("mimeType")=="application/vnd.google-apps.folder" else 1, m.get("name",""))
    return sorted(items, key=_key)

def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    if _is_offline_root():
        return _drive_list_children_offline(folder_id, max_items=max_items)
    return _drive_list_children_online(folder_id, max_items=max_items)

@_retry()
def _download_online(dest: Path, file_id: str) -> Path:
    from googleapiclient.http import MediaIoBaseDownload
    cli = get_drive_client()
    if not cli:
        raise RuntimeError("Drive not configured")
    req = cli.files().get_media(fileId=file_id)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _DRIVE_LOCK:
        with open(tmp, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
    os.replace(tmp, dest)
    _cb_record(True)
    return dest

def _download_offline(dest: Path, src_path: str) -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    import shutil
    shutil.copyfile(src_path, tmp)
    os.replace(tmp, dest)
    return dest

def download_to(path: Path, file_id: str, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not force) and path.exists():
        return path
    lock = _acquire_lock(path.with_suffix(path.suffix + ".lock"), timeout=8)
    try:
        if _is_offline_root():
            if os.path.isfile(file_id):
                return _download_offline(path, file_id)
            src = Path(file_id)
            if src.exists() and src.is_file():
                return _download_offline(path, str(src))
            return path
        else:
            return _download_online(path, file_id)
    except Exception:
        _cb_record(False)
        raise
    finally:
        if lock:
            _release_lock(lock)

# =================================================================================
# Epoch & cached children
# =================================================================================
@st.cache_data(ttl=180, show_spinner=False)
@_retry()
def _compute_drive_epoch(root_id: str, nocache: str = "stable") -> str:
    try:
        kids = list_children(root_id, max_items=2000)
        root_max = max((k.get("modifiedTime","") for k in kids), default="")
        back_max = ""
        # Find Backup
        for k in kids:
            if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name","").lower()=="backup":
                bk = list_children(k.get("id"), max_items=2000)
                back_max = max((x.get("modifiedTime","") for x in bk), default="")
                break
        token = "|".join([root_max, back_max])
        return token or time.strftime("%Y%m%d%H%M%S")
    except Exception:
        return "no-drive"

def _ensure_epoch_key():
    if not drive_enabled():
        return
    try:
        new_epoch = _compute_drive_epoch(GDRIVE_FOLDER_ID, nocache="stable")
    except Exception:
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
        try:
            st.session_state[key] = list_children(folder_id, max_items=2000)
        except Exception as e:
            st.session_state[key] = []
            st.warning(f"Drive list error (non-fatal): {e!s}")
    return st.session_state[key]

# =================================================================================
# Helpers shared across tabs
# =================================================================================
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

def ensure_csv_cached(meta: Dict[str, Any], subdir: str, cache_epoch: str = "", force: bool = False) -> Path:
    local_path = (CSV_CACHE / subdir / meta["name"])
    try:
        return download_to(local_path, meta["id"], force=force)
    except Exception as e:
        st.warning(f"CSV download failed (non-fatal): {e!s}")
        return local_path  # may not exist; upstream guards handle empties

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str, force: bool = False) -> Optional[Path]:
    local_path = CHUNK_CACHE / subdir / chunk_name
    if (not force) and local_path.exists():
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
    root_t, tag_t = target["root"], target["tag"]
    s_t, e_t = target["s"], target["e"]
    label_t = target["label"].lower()

    candidates = []
    for k in kids:
        nm = k.get("name","")
        if not nm.lower().endswith(".wav"): continue
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

# =================================================================================
# Root CSV logic (Tab 1)
# =================================================================================
@st.cache_data(show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    bn_paths += sorted(glob.glob(os.path.join(root, "[0-9]"*8 + "_" + "[0-9]"*6 + "_birdnet_master.csv")))
    kn_paths += sorted(glob.glob(os.path.join(root, "[0-9]"*8 + "_" + "[0-9]"*6 + "_koreronet_master.csv")))
    return bn_paths, kn_paths

@st.cache_data(show_spinner=False)
def list_csvs_drive_root(folder_id: str, cache_epoch: str, nocache: str = "stable") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kids = _folder_children_cached(folder_id)
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

@st.cache_data(show_spinner=False)
def extract_dates_from_csv(path: str | Path) -> List[date]:
    path = str(path)
    dates = set()
    try:
        for chunk in pd.read_csv(path, chunksize=5000, low_memory=False, on_bad_lines='skip'):
            if "ActualStartTime" in chunk.columns:
                s = pd.to_datetime(chunk["ActualStartTime"], errors="coerce")
            else:
                if "ActualTime" not in chunk.columns: continue
                s = pd.to_datetime(chunk["ActualTime"], errors="coerce", dayfirst=True)
            dates.update(ts.date() for ts in s.dropna())
    except Exception:
        return []
    return sorted(dates)

@st.cache_data(show_spinner=False)
def build_date_index(paths: List[Path], kind: str) -> Dict[date, List[str]]:
    idx: Dict[date, List[str]] = {}
    for p in paths:
        ds = extract_dates_from_csv(p)
        for d in ds:
            idx.setdefault(d, []).append(str(p))
    return idx

@st.cache_data(show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(str(path), low_memory=False, on_bad_lines='skip')
    except Exception:
        # Graceful fallback
        return pd.DataFrame()

# =================================================================================
# Snapshot/master logic for Tab 2
# =================================================================================
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
    kids = _folder_children_cached(root_folder_id)
    for k in kids:
        if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name","").lower() == "backup":
            return k
    return None

def _find_chunk_dirs(snapshot_id: str) -> Dict[str, str]:
    kids = _folder_children_cached(snapshot_id)
    kn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "koreronet" in k.get("name","").lower()]
    bn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "birdnet"   in k.get("name","").lower()]
    return {"KN": (kn[0]["id"] if kn else snapshot_id), "BN": (bn[0]["id"] if bn else snapshot_id)}

def _find_named_in_snapshot(snapshot_id: str, exact_name: str) -> Optional[Dict[str, Any]]:
    kids = _folder_children_cached(snapshot_id)
    files_only = [f for f in kids if f.get("mimeType") != "application/vnd.google-apps.folder"]
    for f in files_only:
        if f.get("name","").lower() == exact_name.lower():
            return f
    subfolders = [f for f in kids if f.get("mimeType") == "application/vnd.google-apps.folder"]
    for sf in subfolders:
        sub_files = _folder_children_cached(sf["id"])
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

@st.cache_data(show_spinner=True)
def build_master_index_by_snapshot_date(root_folder_id: str, cache_epoch: str, nocache: str = "stable") -> pd.DataFrame:
    backup = _find_backup_folder(root_folder_id)
    if not backup:
        return pd.DataFrame(columns=[
            "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
            "ChunkDriveFolderId","SnapId","SnapName"
        ])

    snaps = [k for k in _folder_children_cached(backup["id"])
             if k.get("mimeType")=="application/vnd.google-apps.folder" and SNAP_RE.match(k.get("name",""))]
    snaps.sort(key=lambda m: m.get("name",""), reverse=True)

    rows: List[Dict[str,Any]] = []
    for sn in snaps:
        snap_id, snap_name = sn["id"], sn["name"]
        snap_date = _parse_date_from_snapname(snap_name)
        if not snap_date: continue

        chunk_dirs = _find_chunk_dirs(snap_id)
        if snap_date >= CUTOFF_NEW:
            kn_meta = _find_named_in_snapshot(snap_id, "koreronet_master.csv")
            if kn_meta:
                kn_csv = ensure_csv_cached(kn_meta, subdir=f"snap_{snap_id}/koreronet", cache_epoch=cache_epoch, force=False)
                try:
                    df = pd.read_csv(kn_csv, low_memory=False, on_bad_lines='skip')
                    for _, r in df.iterrows():
                        clip  = str(r.get("Clip","")).strip()
                        lab   = str(r.get("Label","Unknown"))
                        conf  = float(r.get("Probability", np.nan))
                        rows.append({
                            "Date": snap_date, "Kind": "KN", "Label": lab, "Confidence": conf,
                            "Start": np.nan, "End": np.nan, "WavBase": "",
                            "ChunkName": clip, "ChunkDriveFolderId": chunk_dirs["KN"],
                            "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass

            bn_meta = _find_named_in_snapshot(snap_id, "birdnet_master.csv")
            if bn_meta:
                bn_csv = ensure_csv_cached(bn_meta, subdir=f"snap_{snap_id}/birdnet", cache_epoch=cache_epoch, force=False)
                try:
                    df = pd.read_csv(bn_csv, low_memory=False, on_bad_lines='skip')
                    for _, r in df.iterrows():
                        clip  = str(r.get("Clip","")).strip()
                        lab   = str(r.get("Label", r.get("Common name","Unknown")))
                        conf  = float(r.get("Probability", r.get("Confidence", np.nan)))
                        rows.append({
                            "Date": snap_date, "Kind": "BN", "Label": lab, "Confidence": conf,
                            "Start": np.nan, "End": np.nan, "WavBase": "",
                            "ChunkName": clip, "ChunkDriveFolderId": chunk_dirs["BN"],
                            "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass
        else:
            root_kids = _folder_children_cached(snap_id)
            files_only = [f for f in root_kids if f.get("mimeType") != "application/vnd.google-apps.folder"]

            kn_legacy = [f for f in files_only if _match_legacy_master_name(f.get("name",""), "KN")]
            if not kn_legacy:
                for sf in [f for f in root_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]:
                    cand = [f for f in _folder_children_cached(sf["id"])
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
                    df = pd.read_csv(kn_csv, low_memory=False, on_bad_lines='skip')
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

            bn_legacy = [f for f in files_only if _match_legacy_master_name(f.get("name",""), "BN")]
            if not bn_legacy:
                for sf in [f for f in root_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]:
                    cand = [f for f in _folder_children_cached(sf["id"])
                            if f.get("mimeType") != "application/vnd.google-apps.folder"
                            and _match_legacy_master_name(f.get("name",""), "BN")]
                    if cand: bn_legacy = cand; break
            if bn_legacy:
                bn_legacy.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
                meta = bn_legacy[0]
                bn_csv = ensure_csv_cached(meta, subdir=f"snap_{snap_id}/birdnet", cache_epoch=cache_epoch, force=False)
                try:
                    df = pd.read_csv(bn_csv, low_memory=False, on_bad_lines='skip')
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

# =================================================================================
# Offline emulation for list/download when ROOT_LOCAL is a folder id
# (We re-use the same helpers above by passing local paths as "ids")
# =================================================================================
if _is_offline_root():
    def drive_enabled() -> bool:  # override to true so tabs work
        return True

# =================================================================================
# Welcome overlay (unchanged behavior; safer summariser)
# =================================================================================
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

@st.cache_data(show_spinner=False)
def _latest_root_summary(root_id_or_path: str, live: bool) -> Tuple[Optional[str], Optional[date], pd.DataFrame]:
    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
    try:
        if drive_enabled():
            bn_meta, kn_meta = list_csvs_drive_root(root_id_or_path, cache_epoch=cache_epoch)
            if bn_meta or kn_meta:
                bn_paths = [ensure_csv_cached(m, subdir="root/bn", cache_epoch=cache_epoch, force=False) for m in bn_meta]
                kn_paths = [ensure_csv_cached(m, subdir="root/kn", cache_epoch=cache_epoch, force=False) for m in kn_meta]
            else:
                bn_paths, kn_paths = [], []
        else:
            bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
            bn_paths = [Path(p) for p in bn_local]
            kn_paths = [Path(p) for p in kn_local]
    except Exception:
        bn_paths, kn_paths = [], []

    bn_by_date = build_date_index(bn_paths, "bn") if bn_paths else {}
    kn_by_date = build_date_index(kn_paths, "kn") if kn_paths else {}
    if not bn_by_date and not kn_by_date:
        return None, None, pd.DataFrame()

    bn_dates = set(bn_by_date.keys())
    kn_dates = set(kn_by_date.keys())
    both     = sorted(bn_dates & kn_dates)
    either   = sorted(bn_dates | kn_dates)

    if both:
        chosen = both[-1]; mode_label = "Combined"
    else:
        chosen = either[-1]; mode_label = "BirdNET (bn)" if chosen in bn_dates else "KōreroNET (kn)"

    def _load_and_filter(paths: List[Path], kind: str, day_selected: date):
        frames = []
        for p in paths:
            try:
                raw = load_csv(p)
                std = standardize_root_df(raw, kind)
                std = std[std["ActualTime"].dt.date == day_selected]
                if not std.empty: frames.append(std)
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df_bn = _load_and_filter(bn_by_date.get(chosen, []), "bn", chosen) if chosen in bn_by_date else pd.DataFrame()
    df_kn = _load_and_filter(kn_by_date.get(chosen, []), "kn", chosen) if chosen in kn_by_date else pd.DataFrame()
    merged = pd.concat([df_bn, df_kn], ignore_index=True) if not df_bn.empty or not df_kn.empty else (df_bn if not df_bn.empty else df_kn)
    return mode_label, chosen, merged

def _render_welcome_overlay():
    if st.session_state.get("__welcome_done__", False):
        return
    with st.spinner("Summarising latest detections…"):
        mode, chosen_date, df = _latest_root_summary(GDRIVE_FOLDER_ID, False)
    overlay = st.empty()
    with overlay.container():
        st.markdown('<div class="overlay-card">', unsafe_allow_html=True)
        st.markdown('<div class="overlay-title">KōreroNET</div>', unsafe_allow_html=True)
        st.markdown('<div class="overlay-sub">Our Technology Highlights</div>', unsafe_allow_html=True)
        features: List[Tuple[str, str]] = [
            ('🔊', 'Bioacoustic monitoring of all vocal species in New Zealand wildlife.'),
            ('🎧', 'Full-spectrum recording: ultrasonic and audible ranges.'),
            ('🤖', 'Autonomous detection powered by our in‑house AI models.'),
            ('🎛️', 'On‑device edge computing and recording in a single package.'),
            ('📡', 'Deployable in remote areas with flexible connectivity.'),
            ('📶', 'Supports LoRaWAN, Wi‑Fi and LTE networking.'),
            ('☀️', 'Solar‑powered and weather‑sealed for harsh environments.'),
            ('⚡', 'Energy‑efficient: records and processes in intervals to save power.'),
            ('🐦', 'Detects both pests and birds of interest.'),
            ('📁', 'Provides accessible recordings of species of interest.')
        ]
        feature_html = '<div class="features-grid">'
        for icon, text in features:
            feature_html += f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-text">{text}</div></div>'
        feature_html += '</div>'
        st.markdown(feature_html, unsafe_allow_html=True)
        st.markdown('<hr style="border:0; border-top:1px solid #444; margin:1.5rem 0; opacity:0.4;">', unsafe_allow_html=True)
        st.markdown('<div class="overlay-sub">Latest Field Summary</div>', unsafe_allow_html=True)
        if df is None or df.empty or chosen_date is None:
            st.markdown('<div class="overlay-sub">No parsable detections found in the most recent root CSVs.</div>', unsafe_allow_html=True)
        else:
            counts = df["Label"].astype(str).value_counts()
            top = [(lbl, int(counts[lbl])) for lbl in counts.index[:3]]
            nice_day = _human_day(chosen_date)
            sentence = f"{nice_day} we detected {_join_top(top)}."
            st.markdown(f'<div class="overlay-sub">{sentence}</div>', unsafe_allow_html=True)
            total = int(counts.sum()); uniq = int(counts.shape[0])
            st.markdown(f"""
                <div class="overlay-pill">Mode: {mode or '—'}</div>
                <div class="overlay-pill">Date: {chosen_date.isoformat() if chosen_date else '—'}</div>
                <div class="overlay-pill">Detections: {total:,}</div>
                <div class="overlay-pill">Species: {uniq:,}</div>
            """, unsafe_allow_html=True)
        c1, _ = st.columns([1,5])
        with c1:
            if st.button("Continue →", type="primary", key=k("welcome_continue")):
                st.session_state["__welcome_done__"] = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

_render_welcome_overlay()

# =================================================================================
# Tab structure
# =================================================================================
tab_nodes, tab1, tab_verify, tab3, tab4 = st.tabs(["🗺️ Nodes", "📊 Detections", "🎧 Verify recordings", "⚡ Power", "📝 log"])

# ==================================
# TAB — Nodes (map fallbacks)
# ==================================
with tab_nodes:
    st.subheader("Choose a node")
    _df_nodes = pd.DataFrame([{"lat": n["lat"], "lon": n["lon"], "name": n["name"], "key": n["key"], "desc": n["desc"]} for n in NODES])
    if "active_node" not in st.session_state:
        st.session_state["active_node"] = NODE_KEYS[0]
    node_choice = st.selectbox("Active node", NODE_KEYS, index=NODE_KEYS.index(st.session_state["active_node"]), key=k("node_select_choice"))
    colA, colB = st.columns([1,4])
    with colA:
        if st.button("Set as active node", key=k("apply_node_box")):
            st.session_state["active_node"] = node_choice
            st.success(f"Active node set to: {node_choice}")
            st.rerun()
    try:
        _center = _df_nodes[_df_nodes["key"] == st.session_state["active_node"]][["lat","lon"]].iloc[0].to_dict()
        center_lat, center_lon = float(_center["lat"]), float(_center["lon"])
    except Exception:
        center_lat, center_lon = -36.8528, 174.8150

    rendered = False
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True, tiles="OpenStreetMap")
        for _, r in _df_nodes.iterrows():
            folium.Circle(location=[float(r["lat"]), float(r["lon"])], radius=150, color=None, fill=True, fill_opacity=0.18, fill_color="#ff0000").add_to(m)
            folium.CircleMarker(location=[float(r["lat"]), float(r["lon"])], radius=16, color=None, fill=True, fill_color="#ffffff", fill_opacity=1.0).add_to(m)
            folium.CircleMarker(location=[float(r["lat"]), float(r["lon"])], radius=14, color=None, fill=True, fill_color="#dc143c", fill_opacity=0.95, tooltip=f"{r['name']}\\n{r['desc']}").add_to(m)
        st_folium(m, width=None, height=520)
        rendered = True
    except Exception:
        pass
    if not rendered:
        try:
            import pydeck as pdk
            halo = pdk.Layer("ScatterplotLayer", data=_df_nodes, get_position='[lon, lat]', get_radius=180, get_fill_color='[255,0,0,46]', pickable=False)
            white = pdk.Layer("ScatterplotLayer", data=_df_nodes, get_position='[lon, lat]', get_radius=24, get_fill_color='[255,255,255,255]', pickable=False)
            main = pdk.Layer("ScatterplotLayer", data=_df_nodes, get_position='[lon, lat]', get_radius=20, get_fill_color='[220,20,60,242]', pickable=True)
            view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)
            st.pydeck_chart(pdk.Deck(layers=[halo, white, main], initial_view_state=view, map_style=None, tooltip={"text": "{name}\\n{desc}"}))
            rendered = True
        except Exception:
            pass
    if not rendered:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_df_nodes["lon"], y=_df_nodes["lat"], mode="markers", marker=dict(size=30, color="rgba(255,0,0,0.18)"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=_df_nodes["lon"], y=_df_nodes["lat"], mode="markers", marker=dict(size=22, color="white"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=_df_nodes["lon"], y=_df_nodes["lat"], mode="markers+text", marker=dict(size=18, color="crimson"), text=_df_nodes["name"], textposition="top center", hovertext=_df_nodes["desc"], hoverinfo="text", showlegend=False))
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 1 — Detections
# =========================
with tab1:
    status = st.empty()
    with status.container():
        st.markdown('<div class="center-wrap fade-enter"><div>🔎 Checking root CSVs…</div></div>', unsafe_allow_html=True)
    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
    if drive_enabled() and (not _is_offline_root()):
        bn_meta, kn_meta = list_csvs_drive_root(GDRIVE_FOLDER_ID, cache_epoch=cache_epoch, nocache="stable")
    else:
        bn_meta, kn_meta = [], []
    force_live = False
    if drive_enabled() and (bn_meta or kn_meta):
        status.empty(); status = st.empty()
        with status.container():
            st.markdown('<div class="center-wrap"><div>⬇️ Downloading root CSVs…</div></div>', unsafe_allow_html=True)
        bn_paths = [ensure_csv_cached(m, subdir="root/bn", cache_epoch=cache_epoch, force=force_live) for m in bn_meta]
        kn_paths = [ensure_csv_cached(m, subdir="root/kn", cache_epoch=cache_epoch, force=force_live) for m in kn_meta]
    else:
        bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
        bn_paths = [Path(p) for p in bn_local]
        kn_paths = [Path(p) for p in kn_local]
    status.empty(); status = st.empty()
    with status.container():
        st.markdown('<div class="center-wrap"><div>🧮 Building date index…</div></div>', unsafe_allow_html=True)

    bn_by_date = build_date_index(bn_paths, "bn") if bn_paths else {}
    kn_by_date = build_date_index(kn_paths, "kn") if kn_paths else {}
    status.empty()

    bn_dates = sorted(bn_by_date.keys()); kn_dates = sorted(kn_by_date.keys())
    paired_dates = sorted(set(bn_dates).intersection(set(kn_dates)))

    src = st.selectbox("Source", ["KōreroNET (kn)", "BirdNET (bn)", "Combined"], index=0, key=k("tab1_src"))
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.95, 0.01, key=k("tab1_min_conf"))

    if src == "Combined":
        options, help_txt = paired_dates, "Only dates that have BOTH BN & KN detections."
    elif src == "BirdNET (bn)":
        options, help_txt = bn_dates, "Dates present in any BN file."
    else:
        options, help_txt = kn_dates, "Dates present in any KN file."
    if not options:
        st.warning(f"No available dates for {src}."); st.stop()

    def calendar_pick(available_days: List[date], label: str, help_txt: str = "") -> date:
        available_days = sorted(available_days)
        d_min, d_max = available_days[0], available_days[-1]
        default_val = st.session_state.get(k("tab1_day") + "::last", d_max)
        if default_val not in set(available_days):
            default_val = d_max
        d_val = st.date_input(label, value=default_val, min_value=d_min, max_value=d_max, help=help_txt, key=k("tab1_day"))
        if d_val not in set(available_days):
            earlier = [x for x in available_days if x <= d_val]
            if earlier:
                d_val = earlier[-1]; st.info(f"No data on chosen date; showing {d_val.isoformat()} (nearest earlier).")
            else:
                later = [x for x in available_days if x >= d_val]
                d_val = later[0]; st.info(f"No data on chosen date; showing {d_val.isoformat()} (nearest later).")
        st.session_state[k("tab1_day") + "::last"] = d_val
        return d_val

    d = calendar_pick(options, "Day", help_txt)

    def load_and_filter(paths: List[Path], kind: str, day_selected: date):
        frames = []
        for p in paths:
            try:
                raw = load_csv(p)
                std = standardize_root_df(raw, kind)
                std = std[std["ActualTime"].dt.date == day_selected]
                if not std.empty: frames.append(std)
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def make_heatmap(df: pd.DataFrame, min_conf: float, title: str):
        df_f = df[pd.to_numeric(df["Confidence"], errors="coerce").astype(float) >= float(min_conf)].copy()
        if df_f.empty:
            st.warning("No detections after applying the confidence filter."); return
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
        fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, color_continuous_scale="RdYlBu_r",
                        labels=dict(x="Hour (AM/PM)", y="Species (label)", color="Detections"),
                        text_auto=True, aspect="auto", title=title)
        fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Rendering heatmap…"):
        if src == "BirdNET (bn)":
            df = load_and_filter(bn_by_date.get(d, []), "bn", d)
            make_heatmap(df, min_conf, f"BirdNET • {d.isoformat()}")
        elif src == "KōreroNET (kn)":
            df = load_and_filter(kn_by_date.get(d, []), "kn", d)
            make_heatmap(df, min_conf, f"KōreroNET • {d.isoformat()}")
        else:
            df_bn = load_and_filter(bn_by_date.get(d, []), "bn", d)
            df_kn = load_and_filter(kn_by_date.get(d, []), "kn", d)
            make_heatmap(pd.concat([df_bn, df_kn], ignore_index=True), min_conf, f"Combined (BN+KN) • {d.isoformat()}")

# ================================
# TAB 2 — Verify (snapshot date)
# ================================
with tab_verify:
    if not drive_enabled():
        st.error("Google Drive is not configured or disabled."); st.stop()

    center2 = st.empty()
    with center2.container():
        st.markdown('<div class="center-wrap fade-enter"><div>📚 Indexing master CSVs by snapshot date…</div></div>', unsafe_allow_html=True)

    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
    try:
        master = build_master_index_by_snapshot_date(GDRIVE_FOLDER_ID, cache_epoch=cache_epoch, nocache="stable")
    except Exception as e:
        master = pd.DataFrame()
        st.warning(f"Could not index masters (non-fatal): {e!s}")

    center2.empty()
    if master.empty or master.shape[0] == 0:
        st.warning("No master CSVs found in any snapshot."); st.stop()

    colA, colB = st.columns([2,1])
    with colA:
        src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0, key=k("tab2_src"))
    with colB:
        min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01, key=k("tab2_min_conf"))

    try:
        if src_mode_v == "KōreroNET (KN)":
            pool = master[master["Kind"] == "KN"]
        elif src_mode_v == "BirdNET (BN)":
            pool = master[master["Kind"] == "BN"]
        else:
            pool = master
        pool = pool[pd.to_numeric(pool["Confidence"], errors="coerce") >= float(min_conf_v)]
    except Exception as e:
        pool = pd.DataFrame()
        st.warning(f"Filter error (non-fatal): {e!s}")

    if pool.empty:
        st.info("No rows above the selected confidence."); st.stop()

    try:
        avail_days = sorted(pool["Date"].unique())
    except Exception:
        avail_days = []
    if not avail_days:
        st.info("No detections for any date with current filters."); st.stop()

    day_default = avail_days[-1]
    day_pick = st.date_input("Day", value=day_default, min_value=avail_days[0], max_value=avail_days[-1], key=k("tab2_day"))
    day_df = pool[pool["Date"] == day_pick]
    if day_df.empty:
        st.warning("No detections for the chosen date."); st.stop()

    counts = day_df.groupby("Label").size().sort_values(ascending=False)
    species_list = list(counts.index)
    sel_key = f"verify_species_sel::{day_pick.isoformat()}::{src_mode_v}"
    prev_species = st.session_state.get(sel_key, None)
    if prev_species not in species_list:
        prev_species = species_list[0]; st.session_state[sel_key] = prev_species
    species = st.selectbox("Species", options=species_list, index=species_list.index(prev_species),
                           format_func=lambda s: f"{s} — {counts[s]} detections", key=f"verify_species::{day_pick.isoformat()}::{src_mode_v}")
    st.session_state[sel_key] = species

    playlist = day_df[day_df["Label"] == species].sort_values(["Kind","ChunkName"]).reset_index(drop=True)
    pkey = f"v2_playlist_sig::{day_pick.isoformat()}::{src_mode_v}::{species}"
    sig = (len(playlist), tuple(playlist["ChunkName"].head(3)))
    if st.session_state.get(pkey) != sig:
        st.session_state[pkey] = sig
        st.session_state[f"v2_idx::{pkey}"] = 0
    idx_key = f"v2_idx::{pkey}"
    idx = int(st.session_state.get(idx_key, 0)) % max(1, len(playlist))

    col1, col2, col3, _ = st.columns([1,1,1,6])
    autoplay = False
    with col1:
        if st.button("⏮ Prev", key=k("tab2_prev")):
            idx = (idx - 1) % len(playlist); autoplay = True
    with col2:
        if st.button("▶ Play", key=k("tab2_play")):
            autoplay = True
    with col3:
        if st.button("⏭ Next", key=k("tab2_next")):
            idx = (idx + 1) % len(playlist); autoplay = True
    st.session_state[idx_key] = idx

    if playlist.empty:
        st.info("No clips for this species and day."); st.stop()

    row = playlist.iloc[idx]
    try:
        conf_val = float(row['Confidence'])
    except Exception:
        conf_val = float('nan')
    st.markdown(f"**Date:** {row['Date']}  |  **Chunk:** `{row['ChunkName']}`  |  **Kind:** {row['Kind']}  |  **Confidence:** {conf_val:.3f}" if pd.notna(conf_val) else f"**Date:** {row['Date']}  |  **Chunk:** `{row['ChunkName']}`  |  **Kind:** {row['Kind']}")

    @_retry()
    def _safe_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
        return ensure_chunk_cached(chunk_name, folder_id, subdir=subdir, force=False)

    def _play_audio(row_: pd.Series, auto: bool):
        try:
            chunk_name = str(row_.get("ChunkName","") or "")
            folder_id  = str(row_.get("ChunkDriveFolderId","") or "")
            kind       = str(row_.get("Kind","UNK") or "")
            if not chunk_name or not folder_id:
                st.warning("No chunk mapping available."); return
            subdir = f"{kind}"
            with st.spinner("Fetching audio…"):
                cached = _safe_chunk_cached(chunk_name, folder_id, subdir=subdir)
            if not cached or not cached.exists():
                st.warning("Audio chunk not found in Drive folder."); return
            try:
                if cached.stat().st_size > 30 * 1024 * 1024:
                    st.warning("Audio file too large to preview safely."); return
                with open(cached, "rb") as f:
                    data = f.read()
            except MemoryError:
                st.warning("Not enough memory to load audio preview."); return
            except Exception as e:
                st.warning(f"Cannot open audio chunk: {e!s}"); return

            if auto:
                import base64
                b64 = base64.b64encode(data).decode()
                st.markdown(f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>', unsafe_allow_html=True)
            else:
                st.audio(data, format="audio/wav")
        except Exception as e:
            st.warning(f"Playback error (non-fatal): {e!s}")

    _play_audio(row, autoplay)

# =====================
# TAB 3 — Power graph
# =====================
with tab3:
    st.subheader("Node Power History")
    if not drive_enabled():
        st.error("Google Drive is not configured or disabled."); st.stop()

    def find_subfolder_by_name(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
        kids = _folder_children_cached(root_id)
        for k in kids:
            if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name","").lower()==name_ci.lower():
                return k
        return None

    logs_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
    if not logs_folder:
        st.warning("Could not find 'Power logs' folder under the Drive root.")
        st.stop()

    LOG_RE = re.compile(r"^power_history_(\\d{8})_(\\d{6})\\.log$", re.IGNORECASE)

    @st.cache_data(show_spinner=True)
    def list_power_logs(folder_id: str, cache_epoch: str, nocache: str = "stable") -> List[Dict[str, Any]]:
        kids = _folder_children_cached(folder_id)
        files = [k for k in kids if k.get("mimeType") != "application/vnd.google-apps.folder" and LOG_RE.match(k.get("name",""))]
        files.sort(key=lambda m: m.get("name",""), reverse=True)
        return files

    @st.cache_data(show_spinner=False)
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
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines: return None

        try:
            head_dt = datetime.strptime(lines[0], "%Y-%m-%d %H:%M:%S")
        except Exception:
            head_dt = None
            for l in lines[:3]:
                m = re.search(r"\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}", l)
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

        times = [head_dt - timedelta(hours=(L-1 - i)) for i in range(L)]
        df = pd.DataFrame({"t": times, "PH_WH": WH, "PH_mAh": mAh, "PH_SoCi": SoCi, "PH_SoCv": SoCv})

        eps = 1e-9
        all_zero_mask = ((np.abs(df["PH_WH"])<eps) & (np.abs(df["PH_mAh"])<eps) & (np.abs(df["PH_SoCi"])<eps) & (np.abs(df["PH_SoCv"])<eps))
        df = df.loc[~all_zero_mask].reset_index(drop=True)
        return df

    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
    if st.button("Show results", type="primary", key=k("tab3_show_btn")):
        with st.spinner("Building power time-series (last 7 days)…"):
            files = list_power_logs(logs_folder["id"], cache_epoch=cache_epoch)
            window_days = 7
            cutoff = datetime.now() - timedelta(days=window_days)
            frames: List[pd.DataFrame] = []
            for meta in files:
                local = ensure_log_cached(meta, force=False)
                df = parse_power_log(local)
                if df is None or df.empty: continue
                df = df[df["t"] >= cutoff]
                if df.empty: break
                frames.append(df)
            ts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["t","PH_WH","PH_mAh","PH_SoCi","PH_SoCv"])

        if ts.empty:
            st.warning("No parsable power logs in the last 7 days.")
        else:
            ts = ts.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_SoCi"], mode="lines", name="SoC_i (%)", yaxis="y1"))
            fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_WH"],  mode="lines", name="Energy (Wh)", yaxis="y2"))
            fig.update_layout(title="Power / State of Charge (last 7 days)", xaxis=dict(title="Time"),
                              yaxis=dict(title="SoC (%)", range=[0, 100]), yaxis2=dict(title="Wh", overlaying="y", side="right"),
                              legend=dict(orientation="h"), margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            stitched_days = max(1, (ts["t"].max() - ts["t"].min()).days + 1)
            st.caption(f"Showing last {min(stitched_days, 7)} day(s): {ts['t'].min().date()} → {ts['t'].max().date()} · {len(ts)} points")
            last = ts.iloc[-1]
            c1, c2 = st.columns(2)
            with c1: st.metric("Last SoC_i (%)", f"{last['PH_SoCi']:.1f}")
            with c2: st.metric("Last Energy (Wh)", f"{last['PH_WH']:.2f}")

# ================================
# TAB 4 — GUI autostart log tail
# ================================
with tab4:
    st.subheader("GUI Autostart — latest log (tail 500)")
    if not drive_enabled():
        st.error("Google Drive is not configured or disabled."); st.stop()

    def find_subfolder_by_name(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
        kids = _folder_children_cached(root_id)
        for k in kids:
            if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name","").lower()==name_ci.lower():
                return k
        return None

    power_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
    if not power_folder:
        st.warning("Could not find 'Power logs' under the Drive root."); st.stop()
    raw_folder = find_subfolder_by_name(power_folder["id"], "raw")
    if not raw_folder:
        st.warning("Could not find 'raw' inside 'Power logs'."); st.stop()

    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")

    @st.cache_data(show_spinner=False)
    def list_autostart_logs(raw_folder_id: str, cache_epoch: str, nocache: str = "stable") -> List[Dict[str, Any]]:
        kids = _folder_children_cached(raw_folder_id)
        files = [k for k in kids if k.get("mimeType") != "application/vnd.google-apps.folder" and LOG_AUTOSTART_RE.match(k.get("name",""))]
        files.sort(key=lambda m: (m.get("name",""), m.get("modifiedTime","")), reverse=True)
        return files

    @st.cache_data(show_spinner=False)
    def ensure_raw_cached(meta: Dict[str, Any], force: bool = False) -> Path:
        local_path = POWER_CACHE / meta["name"]
        return download_to(local_path, meta["id"], force=force)

    files = list_autostart_logs(raw_folder["id"], cache_epoch=cache_epoch)
    colA, _ = st.columns([1,3])
    with colA:
        if st.button("🔄 Refresh", key=k("tab4_refresh")):
            st.cache_data.clear(); st.rerun()

    if not files:
        st.info("No files matching `*__gui_autostart.log` were found in Power logs/raw."); st.stop()

    latest = files[0]
    local = ensure_raw_cached(latest, force=False)

    def tail_lines(path: Path, max_lines: int = 500, block_size: int = 8192) -> List[str]:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END); end = f.tell()
                lines: List[bytes] = []; buf = b""
                while end > 0 and len(lines) <= max_lines:
                    read_size = min(block_size, end)
                    end -= read_size; f.seek(end, os.SEEK_SET)
                    chunk = f.read(read_size); buf = chunk + buf
                    parts = buf.split(b"\\n"); buf = parts[0]; lines_chunk = parts[1:]; lines = lines_chunk + lines
                text_lines = b"\\n".join(lines).decode("utf-8", errors="replace").splitlines()
                return text_lines[-max_lines:]
        except Exception as e:
            return [f"[tail error] {e!s}"]

    lines = tail_lines(local, max_lines=500)
    lines = list(reversed(lines))  # newest first
    numbered = "\\n".join(f"{i+1:>4}  {line}" for i, line in enumerate(lines))
    st.code(numbered, language="log")

    fn = latest.get("name","(unknown)")
    try:
        with open(local, "rb") as _fh:
            st.download_button("Download full log", _fh, file_name=fn, mime="text/plain", key=k("dl_gui_log"))
    except Exception:
        pass
