#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KōreroNET Dashboard v2
──────────────────────
Bioacoustic monitoring dashboard for New Zealand wildlife.

Tabs:
  1. Nodes    — Field node map + selection
  2. Detect   — Root CSV heatmaps (Drive or local) with calendar + min confidence
  3. Verify   — Audio chunk playback from snapshot backups
  4. Power    — SoC / Wh dual-axis time-series from power logs
  5. Log      — GUI autostart log tail (newest 500 lines)

Performance notes:
  • Drive client is cached with @st.cache_resource (singleton, no re-auth).
  • All data caches use TTL + max_entries to cap memory.
  • Drive API calls request only needed fields.
  • Tabs load data lazily via st.fragment (partial reruns).
  • Welcome summary is non-blocking (inline card, no overlay gate).
"""

from __future__ import annotations

import functools
import io
import json
import os
import random
import re
import glob
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG — must be first Streamlit call                  ║
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="KōreroNET — Bioacoustic Dashboard",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  GLOBAL CSS — clean, modern, functional                      ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
/* ── Import a distinctive font pair ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --kn-green:    #22c55e;
    --kn-green-d:  #16a34a;
    --kn-green-bg: rgba(34,197,94,.08);
    --kn-surface:  rgba(255,255,255,.03);
    --kn-border:   rgba(255,255,255,.07);
    --kn-text:     rgba(255,255,255,.92);
    --kn-muted:    rgba(255,255,255,.55);
    --kn-radius:   12px;
    --kn-font:     'DM Sans', system-ui, sans-serif;
    --kn-mono:     'JetBrains Mono', ui-monospace, monospace;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--kn-font) !important;
}
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1280px; }

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"]       { background: transparent !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
#MainMenu, footer, header      { visibility: hidden; }

/* ── Tabs ── */
.stTabs [role="tablist"]       { gap: .35rem; border-bottom: 1px solid var(--kn-border); padding-bottom: .5rem; }
.stTabs [role="tab"]           { font-family: var(--kn-font); font-weight: 600; font-size: .95rem;
                                 padding: .55rem 1.1rem; border-radius: 8px;
                                 border: 1px solid transparent; transition: all .15s ease; }
.stTabs [role="tab"]:hover     { background: var(--kn-green-bg); }
.stTabs [role="tab"][aria-selected="true"] {
    background: var(--kn-green-bg); border-color: var(--kn-green);
    color: var(--kn-green) !important;
}

/* ── Cards ── */
.kn-card {
    background: var(--kn-surface); border: 1px solid var(--kn-border);
    border-radius: var(--kn-radius); padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.kn-card-header { font-size: 1.05rem; font-weight: 700; margin-bottom: .5rem; color: var(--kn-text); }
.kn-card-body   { font-size: .92rem; line-height: 1.55; color: var(--kn-muted); }

/* ── Hero / Brand ── */
.kn-hero {
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: .25rem;
}
.kn-hero-title {
    font-size: clamp(26px, 4vw, 38px); font-weight: 800;
    letter-spacing: -.01em; line-height: 1.1;
}
.kn-hero-sub {
    font-size: .95rem; color: var(--kn-muted); font-weight: 400;
}
.kn-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--kn-green);
    box-shadow: 0 0 0 0 rgba(34,197,94,.6);
    animation: kn-pulse 2s infinite;
    display: inline-block; vertical-align: middle; margin-left: .5rem;
}
@keyframes kn-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,.55); }
    70%  { box-shadow: 0 0 0 14px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

/* ── Summary pills ── */
.kn-pills { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .65rem; }
.kn-pill  {
    display: inline-flex; align-items: center; gap: .3rem;
    padding: .3rem .7rem; border-radius: 999px; font-size: .82rem; font-weight: 600;
    background: var(--kn-surface); border: 1px solid var(--kn-border);
    color: var(--kn-muted);
}
.kn-pill-green { border-color: rgba(34,197,94,.25); color: var(--kn-green); background: var(--kn-green-bg); }

/* ── Feature grid (welcome) ── */
.kn-features {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: .75rem; margin: 1rem 0;
}
.kn-feat {
    display: flex; align-items: flex-start; gap: .6rem;
    background: var(--kn-surface); border: 1px solid var(--kn-border);
    border-radius: 10px; padding: .7rem .85rem;
}
.kn-feat-icon { font-size: 1.2rem; line-height: 1.35rem; flex-shrink: 0; }
.kn-feat-text { font-size: .88rem; line-height: 1.4; color: var(--kn-muted); }

/* ── Metric row ── */
.kn-metrics { display: flex; gap: 1rem; flex-wrap: wrap; margin: .75rem 0; }
.kn-metric  {
    flex: 1; min-width: 140px; padding: .85rem 1rem;
    background: var(--kn-surface); border: 1px solid var(--kn-border);
    border-radius: 10px; text-align: center;
}
.kn-metric-val   { font-size: 1.5rem; font-weight: 800; color: var(--kn-green); }
.kn-metric-label { font-size: .78rem; color: var(--kn-muted); margin-top: .15rem; }

/* ── Player card ── */
.kn-player {
    background: var(--kn-surface); border: 1px solid var(--kn-border);
    border-radius: var(--kn-radius); padding: 1rem 1.25rem;
    margin: .75rem 0;
}
.kn-player-meta { font-family: var(--kn-mono); font-size: .82rem; color: var(--kn-muted); line-height: 1.6; }

/* ── Plots: give them breathing room ── */
.stPlotlyChart { border-radius: var(--kn-radius); overflow: hidden; }

/* ── Code blocks ── */
code, .stCode, [data-testid="stCode"] { font-family: var(--kn-mono) !important; }

/* ── Streamlit form tweaks ── */
[data-testid="stForm"] { border: 1px solid var(--kn-border) !important; border-radius: var(--kn-radius) !important; }

/* ── Small helper ── */
.kn-muted { color: var(--kn-muted); font-size: .88rem; }
.kn-divider { border: 0; border-top: 1px solid var(--kn-border); margin: 1.25rem 0; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONSTANTS                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

# Regex patterns
CHUNK_RE = re.compile(
    r"^(?P<root>\d{8}_\d{6})__(?P<tag>bn|kn)_(?P<s>\d+\.\d{2})_(?P<e>\d+\.\d{2})__(?P<label>.+?)__p(?P<conf>\d+\.\d{2})\.wav$",
    re.IGNORECASE,
)
NEW_ROOT_BN = re.compile(r"^\d{8}_\d{6}_birdnet_master\.csv$", re.IGNORECASE)
NEW_ROOT_KN = re.compile(r"^\d{8}_\d{6}_koreronet_master\.csv$", re.IGNORECASE)
SNAP_RE     = re.compile(r"^(\d{8})_(\d{6})$")
LOG_RE      = re.compile(r"^power_history_(\d{8})_(\d{6})\.log$", re.IGNORECASE)
LOG_AUTOSTART_RE = re.compile(r"^(\d{8})_(\d{6})__gui_autostart\.log$", re.IGNORECASE)
CUTOFF_NEW  = date(2025, 10, 31)

NODES = [
    {"key": "Auckland-Sunnyhills", "name": "Auckland — Sunnyhills", "lat": -36.9003, "lon": 174.8839,
     "desc": "Field node in Sunnyhills, Auckland"},
]
NODE_KEYS = [n["key"] for n in NODES]

# File caches
CACHE_ROOT  = Path("/tmp/koreronet_cache")
CSV_CACHE   = CACHE_ROOT / "csv"
CHUNK_CACHE = CACHE_ROOT / "chunks"
POWER_CACHE = CACHE_ROOT / "power"
for _p in (CSV_CACHE, CHUNK_CACHE, POWER_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# Offline / online
OFFLINE_DEPLOY: bool = False
DEFAULT_ROOT = r"G:\My Drive\From the node"
ROOT_LOCAL   = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)

DRIVE_FIELDS = "nextPageToken, files(id,name,mimeType,modifiedTime,size)"


# ╔══════════════════════════════════════════════════════════════╗
# ║  UTILITIES                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def _retry(exceptions=(Exception,), tries=3, base_delay=0.4, jitter=0.25):
    """Retry with jittered backoff. Avoids catching Streamlit internal exceptions."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            remaining = tries
            delay = base_delay
            while remaining > 0:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    # Never swallow Streamlit control-flow exceptions
                    if type(exc).__name__ in ("RerunException", "StopException"):
                        raise
                    remaining -= 1
                    if remaining <= 0:
                        raise
                    time.sleep(delay + random.random() * jitter)
                    delay = min(1.5, delay * 1.5)
        return wrapper
    return deco


if "_sess_salt" not in st.session_state:
    st.session_state["_sess_salt"] = uuid.uuid4().hex[:8]

def k(name: str) -> str:
    """Deterministic widget key."""
    return f"{name}::{st.session_state['_sess_salt']}"


def _secret_or_env(name: str, default=None):
    """Read from env, then Streamlit secrets."""
    v = os.getenv(name)
    if v is not None:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return default


GDRIVE_FOLDER_ID = ROOT_LOCAL if OFFLINE_DEPLOY else _secret_or_env("GDRIVE_FOLDER_ID")


# ╔══════════════════════════════════════════════════════════════╗
# ║  GOOGLE DRIVE CLIENT (singleton via cache_resource)          ║
# ╚══════════════════════════════════════════════════════════════╝

def _normalize_pk(pk: str) -> str:
    if not isinstance(pk, str):
        return pk
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    for tag in ("-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"):
        if tag in pk and f"{tag}\n" not in pk and f"\n{tag}" not in pk:
            pk = pk.replace(tag, f"\n{tag}\n")
    return pk.strip()


@st.cache_resource(show_spinner=False)
def _build_drive_client():
    """Build and cache the Drive v3 service. Called once per app lifecycle."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        sa_tbl  = st.secrets.get("service_account")
        sa_json = st.secrets.get("SERVICE_ACCOUNT_JSON")
        if not sa_tbl and not sa_json:
            return None
        info = dict(sa_tbl) if sa_tbl else json.loads(sa_json)
        if "private_key" in info:
            info["private_key"] = _normalize_pk(info["private_key"])
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None


def get_drive():
    if OFFLINE_DEPLOY:
        return "offline"
    return _build_drive_client()


def drive_ok() -> bool:
    return bool(GDRIVE_FOLDER_ID and get_drive())


# ╔══════════════════════════════════════════════════════════════╗
# ║  DRIVE HELPERS                                               ║
# ╚══════════════════════════════════════════════════════════════╝

@_retry()
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    drive = get_drive()
    if not drive or drive == "offline":
        return _list_children_local(folder_id, max_items)
    items: list = []
    token = None
    while len(items) < max_items:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields=DRIVE_FIELDS,
            pageSize=min(100, max_items - len(items)),
            pageToken=token,
            orderBy="folder,name_natural",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return items


def _list_children_local(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    """Offline fallback: read from local filesystem."""
    base = Path(folder_id)
    if not base.is_dir():
        return []
    items = []
    for i, entry in enumerate(sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name))):
        if i >= max_items:
            break
        mt = datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
        items.append({
            "id": str(entry), "name": entry.name,
            "mimeType": "application/vnd.google-apps.folder" if entry.is_dir() else "application/octet-stream",
            "modifiedTime": mt, "size": str(entry.stat().st_size),
        })
    return items


@_retry()
def download_to(path: Path, file_id: str, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.stat().st_size > 0:
        return path
    drive = get_drive()
    if drive == "offline" or OFFLINE_DEPLOY:
        src = Path(file_id)
        if src.is_file():
            import shutil
            shutil.copyfile(src, path)
        return path
    from googleapiclient.http import MediaIoBaseDownload
    req = drive.files().get_media(fileId=file_id)
    with open(path, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return path


@st.cache_data(ttl=300, max_entries=50, show_spinner=False)
def _cached_children(folder_id: str, _epoch: str = "") -> List[Dict[str, Any]]:
    return list_children(folder_id)


def find_subfolder(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
    for kid in _cached_children(root_id, _epoch=_get_epoch()):
        if kid.get("mimeType") == "application/vnd.google-apps.folder" and kid.get("name", "").lower() == name_ci.lower():
            return kid
    return None


def _get_epoch() -> str:
    return st.session_state.get("DRIVE_EPOCH", "0")


def _refresh_epoch():
    """Compute a lightweight freshness token from root + backup modifiedTimes."""
    if not drive_ok():
        return
    try:
        root_kids = list_children(GDRIVE_FOLDER_ID, max_items=500)
        max_mt = max((k.get("modifiedTime", "") for k in root_kids), default="")
        bk = next((k for k in root_kids if k.get("mimeType") == "application/vnd.google-apps.folder"
                    and k.get("name", "").lower() == "backup"), None)
        bk_mt = ""
        if bk:
            bk_kids = list_children(bk["id"], max_items=500)
            bk_mt = max((k.get("modifiedTime", "") for k in bk_kids), default="")
        new_epoch = f"{max_mt}|{bk_mt}" or time.strftime("%Y%m%d%H%M%S")
    except Exception:
        return
    old = st.session_state.get("DRIVE_EPOCH")
    if old and new_epoch != old:
        st.cache_data.clear()
    st.session_state["DRIVE_EPOCH"] = new_epoch


# Run once on startup
if "DRIVE_EPOCH" not in st.session_state:
    _refresh_epoch()


# ╔══════════════════════════════════════════════════════════════╗
# ║  CSV HELPERS                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def ensure_csv_cached(meta: Dict[str, Any], subdir: str, force: bool = False) -> Path:
    return download_to(CSV_CACHE / subdir / meta["name"], meta["id"], force=force)


@st.cache_data(ttl=300, max_entries=30, show_spinner=False)
def list_csvs_drive_root(folder_id: str, _epoch: str = "") -> Tuple[List[Dict], List[Dict]]:
    kids = list_children(folder_id, max_items=2000)
    bn, kn = [], []
    for kid in kids:
        n = kid.get("name", "")
        nl = n.lower()
        if not nl.endswith(".csv"):
            continue
        if nl.startswith("bn") or NEW_ROOT_BN.match(n):
            bn.append(kid)
        elif nl.startswith("kn") or NEW_ROOT_KN.match(n):
            kn.append(kid)
    bn.sort(key=lambda m: m["name"])
    kn.sort(key=lambda m: m["name"])
    return bn, kn


@st.cache_data(ttl=300, max_entries=20, show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    bn += sorted(glob.glob(os.path.join(root, "[0-9]" * 8 + "_" + "[0-9]" * 6 + "_birdnet_master.csv")))
    kn += sorted(glob.glob(os.path.join(root, "[0-9]" * 8 + "_" + "[0-9]" * 6 + "_koreronet_master.csv")))
    return bn, kn


def standardize_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    cols_lower = {c.lower() for c in df.columns}
    if {"clip", "actualstarttime", "label", "probability"} <= cols_lower:
        out = pd.DataFrame()
        out["Label"]      = df.get("Label", "Unknown")
        out["Confidence"]  = pd.to_numeric(df.get("Probability", np.nan), errors="coerce")
        out["ActualTime"]  = pd.to_datetime(df.get("ActualStartTime", pd.NaT), errors="coerce")
        return out.dropna(subset=["Label", "ActualTime"])

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
            possible = [c for c in df.columns if c.lower() in ("label", "common name", "common_name", "species", "class")]
            df["Label"] = df[possible[0]] if possible else "Unknown"
    df["ActualTime"] = pd.to_datetime(df.get("ActualTime", pd.NaT), errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Label", "ActualTime"])
    return df[["Label", "Confidence", "ActualTime"]]


@st.cache_data(ttl=600, max_entries=40, show_spinner=False)
def extract_dates(path: str) -> List[date]:
    dates = set()
    try:
        for chunk in pd.read_csv(path, chunksize=5000, usecols=lambda c: c in ("ActualStartTime", "ActualTime")):
            col = "ActualStartTime" if "ActualStartTime" in chunk.columns else "ActualTime"
            if col not in chunk.columns:
                continue
            s = pd.to_datetime(chunk[col], errors="coerce", dayfirst=(col == "ActualTime"))
            dates.update(ts.date() for ts in s.dropna())
    except Exception:
        pass
    return sorted(dates)


@st.cache_data(ttl=600, max_entries=20, show_spinner=False)
def build_date_index(paths: List[str], kind: str) -> Dict[date, List[str]]:
    idx: Dict[date, List[str]] = {}
    for p in paths:
        for d in extract_dates(str(p)):
            idx.setdefault(d, []).append(str(p))
    return idx


@st.cache_data(ttl=300, max_entries=30, show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ╔══════════════════════════════════════════════════════════════╗
# ║  CHUNK FILENAME HELPERS                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def _parse_chunk(name: str) -> Optional[Dict[str, Any]]:
    m = CHUNK_RE.match(name or "")
    if not m:
        return None
    return {
        "root": m.group("root"), "tag": m.group("tag").lower(),
        "s": float(m.group("s")), "e": float(m.group("e")),
        "label": m.group("label"), "conf": float(m.group("conf")),
    }


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s or "")).strip("_")


def _compose_chunk_name(kind: str, wav_base: str, start: float, end: float, label: str, conf: float) -> str:
    root = Path(wav_base).stem
    tag = "bn" if kind.lower() == "bn" else "kn"
    return f"{root}__{tag}_{start:06.2f}_{end:06.2f}__{_sanitize(label)}__p{conf:.2f}.wav"


# ╔══════════════════════════════════════════════════════════════╗
# ║  CHUNK FETCH (on-demand, with fuzzy match)                   ║
# ╚══════════════════════════════════════════════════════════════╝

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str, force: bool = False) -> Optional[Path]:
    local_path = CHUNK_CACHE / subdir / chunk_name
    if not force and local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    kids = _cached_children(folder_id, _epoch=_get_epoch())
    name_to_id = {kid["name"]: kid["id"] for kid in kids}

    # Exact match
    if chunk_name in name_to_id:
        try:
            download_to(local_path, name_to_id[chunk_name], force=force)
            return local_path
        except Exception:
            return None

    # Fuzzy match
    target = _parse_chunk(chunk_name)
    if not target:
        return None

    candidates = []
    for kid in kids:
        nm = kid.get("name", "")
        if not nm.lower().endswith(".wav"):
            continue
        info = _parse_chunk(nm)
        if not info:
            continue
        if info["root"] == target["root"] and info["tag"] == target["tag"] and info["label"].lower() == target["label"].lower():
            candidates.append((info, kid["id"], nm))

    if not candidates:
        return None

    tol = 0.75
    def score(ci):
        contains = (ci["s"] - tol) <= target["s"] and (ci["e"] + tol) >= target["e"]
        cen = abs((ci["s"] + ci["e"]) / 2 - (target["s"] + target["e"]) / 2)
        return (0 if contains else 1, cen)

    candidates.sort(key=lambda x: score(x[0]))
    _, best_id, best_name = candidates[0]
    try:
        download_to(local_path, best_id, force=force)
        if best_name != chunk_name:
            st.caption(f"⚠️ Fuzzy match: `{chunk_name}` → `{best_name}`")
        return local_path
    except Exception:
        return None


# ╔══════════════════════════════════════════════════════════════╗
# ║  SNAPSHOT / MASTER INDEX (Tab 2)                             ║
# ╚══════════════════════════════════════════════════════════════╝

def _find_backup_folder(root_id: str) -> Optional[Dict[str, Any]]:
    return find_subfolder(root_id, "backup")


def _find_chunk_dirs(snap_id: str) -> Dict[str, str]:
    kids = _cached_children(snap_id, _epoch=_get_epoch())
    kn_dir = next((kid["id"] for kid in kids if kid.get("mimeType") == "application/vnd.google-apps.folder"
                   and "koreronet" in kid.get("name", "").lower()), snap_id)
    bn_dir = next((kid["id"] for kid in kids if kid.get("mimeType") == "application/vnd.google-apps.folder"
                   and "birdnet" in kid.get("name", "").lower()), snap_id)
    return {"KN": kn_dir, "BN": bn_dir}


def _find_file_in_snap(snap_id: str, exact: str) -> Optional[Dict[str, Any]]:
    kids = _cached_children(snap_id, _epoch=_get_epoch())
    files = [f for f in kids if f.get("mimeType") != "application/vnd.google-apps.folder"]
    for f in files:
        if f["name"].lower() == exact.lower():
            return f
    for sf in (f for f in kids if f.get("mimeType") == "application/vnd.google-apps.folder"):
        sub = _cached_children(sf["id"], _epoch=_get_epoch())
        for f in sub:
            if f.get("mimeType") != "application/vnd.google-apps.folder" and f["name"].lower() == exact.lower():
                return f
    return None


def _match_legacy_master(name: str, kind: str) -> bool:
    n = name.lower()
    pat = "koreronet" if kind == "KN" else "birdnet"
    return pat in n and "detect" in n and n.endswith(".csv")


@st.cache_data(ttl=300, max_entries=5, show_spinner=False)
def build_master_index(root_id: str, _epoch: str = "") -> pd.DataFrame:
    """Build a full master index from Backup/YYYYMMDD_HHMMSS snapshots."""
    cols = ["Date", "Kind", "Label", "Confidence", "Start", "End", "WavBase", "ChunkName",
            "ChunkDriveFolderId", "SnapId", "SnapName"]

    if OFFLINE_DEPLOY:
        return _build_master_local(root_id)

    backup = _find_backup_folder(root_id)
    if not backup:
        return pd.DataFrame(columns=cols)

    snaps = [kid for kid in _cached_children(backup["id"], _epoch=_epoch)
             if kid.get("mimeType") == "application/vnd.google-apps.folder" and SNAP_RE.match(kid.get("name", ""))]
    snaps.sort(key=lambda m: m["name"], reverse=True)

    rows: List[Dict[str, Any]] = []
    for sn in snaps:
        snap_id, snap_name = sn["id"], sn["name"]
        m = SNAP_RE.match(snap_name)
        if not m:
            continue
        try:
            snap_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            continue

        chunk_dirs = _find_chunk_dirs(snap_id)

        if snap_date >= CUTOFF_NEW:
            # New format
            for kind_key, csv_name in [("KN", "koreronet_master.csv"), ("BN", "birdnet_master.csv")]:
                meta = _find_file_in_snap(snap_id, csv_name)
                if not meta:
                    continue
                csv_path = ensure_csv_cached(meta, subdir=f"snap_{snap_id}/{kind_key.lower()}")
                try:
                    df = pd.read_csv(csv_path)
                    for _, r in df.iterrows():
                        rows.append({
                            "Date": snap_date, "Kind": kind_key,
                            "Label": str(r.get("Label", r.get("Common name", "Unknown"))),
                            "Confidence": float(r.get("Probability", r.get("Confidence", np.nan))),
                            "Start": np.nan, "End": np.nan, "WavBase": "",
                            "ChunkName": str(r.get("Clip", "")),
                            "ChunkDriveFolderId": chunk_dirs[kind_key],
                            "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass
        else:
            # Legacy format
            for kind_key in ("KN", "BN"):
                all_kids = _cached_children(snap_id, _epoch=_epoch)
                files_only = [f for f in all_kids if f.get("mimeType") != "application/vnd.google-apps.folder"]
                legacy = [f for f in files_only if _match_legacy_master(f.get("name", ""), kind_key)]
                if not legacy:
                    for sf in (f for f in all_kids if f.get("mimeType") == "application/vnd.google-apps.folder"):
                        cand = [f for f in _cached_children(sf["id"], _epoch=_epoch)
                                if f.get("mimeType") != "application/vnd.google-apps.folder"
                                and _match_legacy_master(f.get("name", ""), kind_key)]
                        if cand:
                            legacy = cand
                            break
                if not legacy:
                    continue
                legacy.sort(key=lambda m: m.get("modifiedTime", ""), reverse=True)
                csv_path = ensure_csv_cached(legacy[0], subdir=f"snap_{snap_id}/{kind_key.lower()}")
                try:
                    df = pd.read_csv(csv_path)
                    for _, r in df.iterrows():
                        wav = os.path.basename(str(r.get("File", "")))
                        start = float(r.get("Start", r.get("Start (s)", np.nan)))
                        end   = float(r.get("End",   r.get("End (s)",   np.nan)))
                        lab_col = "Common name" if kind_key == "BN" and "Common name" in df.columns else "Label"
                        lab   = str(r.get(lab_col, "Unknown"))
                        conf  = float(r.get("Confidence", np.nan))
                        rows.append({
                            "Date": snap_date, "Kind": kind_key, "Label": lab, "Confidence": conf,
                            "Start": start, "End": end, "WavBase": wav,
                            "ChunkName": _compose_chunk_name(kind_key.lower(), wav, start, end, lab, conf),
                            "ChunkDriveFolderId": chunk_dirs[kind_key],
                            "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out.sort_values(["Date", "Kind", "Label"], ascending=[False, True, True], inplace=True)
    return out.reset_index(drop=True)


def _build_master_local(root_folder: str) -> pd.DataFrame:
    """Offline version using local filesystem."""
    cols = ["Date", "Kind", "Label", "Confidence", "Start", "End", "WavBase", "ChunkName",
            "ChunkDriveFolderId", "SnapId", "SnapName"]
    backup_path = Path(root_folder) / "Backup"
    if not backup_path.exists():
        return pd.DataFrame(columns=cols)

    snaps = sorted([p for p in backup_path.iterdir() if p.is_dir() and SNAP_RE.match(p.name)],
                   key=lambda p: p.name, reverse=True)
    rows: List[Dict[str, Any]] = []
    for sn in snaps:
        m = SNAP_RE.match(sn.name)
        if not m:
            continue
        try:
            snap_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            continue

        kn_dir = next((p for p in sn.iterdir() if p.is_dir() and "koreronet" in p.name.lower()), sn)
        bn_dir = next((p for p in sn.iterdir() if p.is_dir() and "birdnet" in p.name.lower()), sn)

        if snap_date >= CUTOFF_NEW:
            for kind_key, csv_name, chunk_dir in [("KN", "koreronet_master.csv", kn_dir),
                                                   ("BN", "birdnet_master.csv", bn_dir)]:
                csv_path = sn / csv_name
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    for _, r in df.iterrows():
                        rows.append({
                            "Date": snap_date, "Kind": kind_key,
                            "Label": str(r.get("Label", r.get("Common name", "Unknown"))),
                            "Confidence": float(r.get("Probability", r.get("Confidence", np.nan))),
                            "Start": np.nan, "End": np.nan, "WavBase": "",
                            "ChunkName": str(r.get("Clip", "")),
                            "ChunkDriveFolderId": str(chunk_dir),
                            "SnapId": str(sn), "SnapName": sn.name,
                        })
                except Exception:
                    pass
        else:
            for kind_key, chunk_dir in [("KN", kn_dir), ("BN", bn_dir)]:
                pat = "koreronet" if kind_key == "KN" else "birdnet"
                cands = [p for p in sn.rglob("*.csv") if pat in p.name.lower() and "detect" in p.name.lower()]
                if not cands:
                    continue
                cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                try:
                    df = pd.read_csv(cands[0])
                    for _, r in df.iterrows():
                        wav = os.path.basename(str(r.get("File", "")))
                        start = float(r.get("Start", r.get("Start (s)", np.nan)))
                        end   = float(r.get("End",   r.get("End (s)",   np.nan)))
                        lab_col = "Common name" if kind_key == "BN" and "Common name" in df.columns else "Label"
                        lab   = str(r.get(lab_col, "Unknown"))
                        conf  = float(r.get("Confidence", np.nan))
                        rows.append({
                            "Date": snap_date, "Kind": kind_key, "Label": lab, "Confidence": conf,
                            "Start": start, "End": end, "WavBase": wav,
                            "ChunkName": _compose_chunk_name(kind_key.lower(), wav, start, end, lab, conf),
                            "ChunkDriveFolderId": str(chunk_dir),
                            "SnapId": str(sn), "SnapName": sn.name,
                        })
                except Exception:
                    pass

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out.sort_values(["Date", "Kind", "Label"], ascending=[False, True, True], inplace=True)
    return out.reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  POWER LOG HELPERS                                           ║
# ╚══════════════════════════════════════════════════════════════╝

@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def list_power_logs(folder_id: str, _epoch: str = "") -> List[Dict[str, Any]]:
    kids = _cached_children(folder_id, _epoch=_epoch)
    files = [kid for kid in kids if kid.get("mimeType") != "application/vnd.google-apps.folder"
             and LOG_RE.match(kid.get("name", ""))]
    files.sort(key=lambda m: m["name"], reverse=True)
    return files


@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def list_autostart_logs(folder_id: str, _epoch: str = "") -> List[Dict[str, Any]]:
    kids = _cached_children(folder_id, _epoch=_epoch)
    files = [kid for kid in kids if kid.get("mimeType") != "application/vnd.google-apps.folder"
             and LOG_AUTOSTART_RE.match(kid.get("name", ""))]
    files.sort(key=lambda m: m["name"], reverse=True)
    return files


def ensure_log_cached(meta: Dict[str, Any], force: bool = False) -> Path:
    return download_to(POWER_CACHE / meta["name"], meta["id"], force=force)


def _parse_float_list(line: str) -> List[float]:
    try:
        payload = line.split(",", 1)[1]
    except (IndexError, ValueError):
        return []
    vals = []
    for tok in payload.strip().split(","):
        tok = tok.strip().rstrip(".")
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def parse_power_log(path: Path) -> Optional[pd.DataFrame]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    head_dt = None
    for line in lines[:3]:
        m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
        if m:
            try:
                head_dt = datetime.strptime(m.group(0), "%Y-%m-%d %H:%M:%S")
                break
            except ValueError:
                pass
    if not head_dt:
        return None

    get_line = lambda prefix: next((l for l in lines if l.upper().startswith(prefix)), "")
    WH   = _parse_float_list(get_line("PH_WH"))
    mAh  = _parse_float_list(get_line("PH_MAH"))
    SoCi = _parse_float_list(get_line("PH_SOCI"))
    SoCv = _parse_float_list(get_line("PH_SOCV"))

    L = max(len(WH), len(mAh), len(SoCi), len(SoCv))
    if L == 0:
        return None

    def pad(arr):
        return [np.nan] * (L - len(arr)) + arr if len(arr) < L else arr[:L]

    times = [head_dt - timedelta(hours=(L - 1 - i)) for i in range(L)]
    df = pd.DataFrame({"t": times, "PH_WH": pad(WH), "PH_mAh": pad(mAh), "PH_SoCi": pad(SoCi), "PH_SoCv": pad(SoCv)})
    # Drop all-zero rows
    eps = 1e-9
    mask = (df["PH_WH"].abs() < eps) & (df["PH_mAh"].abs() < eps) & (df["PH_SoCi"].abs() < eps) & (df["PH_SoCv"].abs() < eps)
    return df[~mask].reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  WELCOME SUMMARY (non-blocking — inline card)                ║
# ╚══════════════════════════════════════════════════════════════╝

def _human_day(d: date) -> str:
    today = datetime.now().date()
    if d == today:
        return "today"
    if d == today - timedelta(days=1):
        return "yesterday"
    return d.strftime("%A, %d %b %Y")


@st.cache_data(ttl=300, max_entries=3, show_spinner=False)
def _latest_summary(root_id: str, _epoch: str = "") -> Tuple[Optional[date], pd.DataFrame]:
    """Cached summary of latest detections for the welcome card."""
    if drive_ok():
        bn_meta, kn_meta = list_csvs_drive_root(root_id, _epoch=_epoch)
        bn_paths = [str(ensure_csv_cached(m, subdir="root/bn")) for m in bn_meta]
        kn_paths = [str(ensure_csv_cached(m, subdir="root/kn")) for m in kn_meta]
    else:
        bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
        bn_paths, kn_paths = list(bn_local), list(kn_local)

    bn_idx = build_date_index(bn_paths, "bn") if bn_paths else {}
    kn_idx = build_date_index(kn_paths, "kn") if kn_paths else {}
    all_dates = sorted(set(bn_idx.keys()) | set(kn_idx.keys()))

    if not all_dates:
        return None, pd.DataFrame()

    latest = all_dates[-1]
    frames = []
    for p in bn_idx.get(latest, []):
        try:
            raw = load_csv(p)
            std = standardize_df(raw, "bn")
            frames.append(std[std["ActualTime"].dt.date == latest])
        except Exception:
            pass
    for p in kn_idx.get(latest, []):
        try:
            raw = load_csv(p)
            std = standardize_df(raw, "kn")
            frames.append(std[std["ActualTime"].dt.date == latest])
        except Exception:
            pass

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return latest, merged


def _render_welcome():
    """Non-blocking welcome: inline hero + latest summary + feature highlights."""

    # ── Hero ──
    st.markdown("""
    <div class="kn-hero">
        <div>
            <div class="kn-hero-title">KōreroNET <span class="kn-dot"></span></div>
            <div class="kn-hero-sub">Bioacoustic monitoring — Auckland University of Technology</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Latest summary (cached computation) ──
    try:
        latest, merged = _latest_summary(GDRIVE_FOLDER_ID, _epoch=_get_epoch())

        if latest is not None and not merged.empty:
            counts = merged["Label"].astype(str).value_counts()
            top3 = [(lbl, int(counts[lbl])) for lbl in counts.index[:3]]
            total = int(counts.sum())
            n_species = int(counts.shape[0])

            top_str = ", ".join(f"**{cnt:,} {lbl}**" for lbl, cnt in top3)
            st.markdown(f"""
            <div class="kn-card">
                <div class="kn-card-header">Latest field summary — {_human_day(latest)}</div>
                <div class="kn-card-body">Top detections: {top_str}</div>
                <div class="kn-pills">
                    <span class="kn-pill kn-pill-green">● Live</span>
                    <span class="kn-pill">{total:,} detections</span>
                    <span class="kn-pill">{n_species} species</span>
                    <span class="kn-pill">{latest.isoformat()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif latest is None:
            st.info("No data available. Check your Drive connection or local data path.")
        else:
            st.info("No detections loaded yet. Explore the **Detections** tab to get started.")
    except Exception as exc:
        st.warning(f"Could not load summary: {exc!s}")

    # ── Feature highlights (all 10 platform capabilities) ──
    features = [
        ("🔊", "Bioacoustic monitoring of all vocal species in New Zealand wildlife"),
        ("🎧", "Full-spectrum recording — ultrasonic and audible ranges"),
        ("🤖", "Autonomous detection powered by in-house AI models"),
        ("🎛️", "On-device edge computing and recording in a single package"),
        ("📡", "Deployable in remote areas with flexible connectivity"),
        ("📶", "Supports LoRaWAN, Wi-Fi and LTE networking"),
        ("☀️", "Solar-powered and weather-sealed for harsh environments"),
        ("⚡", "Energy-efficient — records and processes in intervals to save power"),
        ("🐦", "Detects both pests and birds of interest"),
        ("📁", "Provides accessible recordings of species of interest"),
    ]
    feat_html = '<div class="kn-features">'
    for icon, text in features:
        feat_html += f'<div class="kn-feat"><div class="kn-feat-icon">{icon}</div><div class="kn-feat-text">{text}</div></div>'
    feat_html += '</div>'
    st.markdown(feat_html, unsafe_allow_html=True)
    st.markdown('<hr class="kn-divider">', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TOP BAR                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

_render_welcome()

col_node, col_refresh = st.columns([4, 1])
with col_node:
    active_node = st.selectbox("Active node", NODE_KEYS, index=0, key=k("node_top"))
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻ Refresh data", key=k("refresh_btn"), use_container_width=True):
        st.cache_data.clear()
        for sk in list(st.session_state.keys()):
            if str(sk).startswith("drive_kids::") or str(sk).startswith("DRIVE_EPOCH"):
                del st.session_state[sk]
        _refresh_epoch()
        st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  TABS                                                        ║
# ╚══════════════════════════════════════════════════════════════╝

tab_nodes, tab_detect, tab_verify, tab_power, tab_log = st.tabs([
    "🗺️ Nodes", "📊 Detections", "🎧 Verify", "⚡ Power", "📋 Log"
])


# ── TAB: Nodes ──────────────────────────────────────────────────
with tab_nodes:
    st.markdown("#### Node locations")
    st.markdown('<p class="kn-muted">Select a node to centre the map. Additional nodes will appear here as the network grows.</p>', unsafe_allow_html=True)

    _df_nodes = pd.DataFrame(
        [{"lat": n["lat"], "lon": n["lon"], "name": n["name"], "key": n["key"], "desc": n["desc"]} for n in NODES]
    )

    try:
        row = _df_nodes[_df_nodes["key"] == active_node].iloc[0]
        clat, clon = float(row["lat"]), float(row["lon"])
    except Exception:
        clat, clon = -36.9003, 174.8839

    # Map fallbacks: folium → pydeck → plotly scatter
    rendered = False
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[clat, clon], zoom_start=13, control_scale=True, tiles="CartoDB Positron")
        for _, r in _df_nodes.iterrows():
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])], radius=10,
                color="#dc143c", fill=True, fill_color="#dc143c", fill_opacity=0.85,
                tooltip=f"<b>{r['name']}</b><br>{r['desc']}",
            ).add_to(m)
        st_folium(m, width=None, height=480, returned_objects=[])
        rendered = True
    except Exception:
        pass

    if not rendered:
        try:
            import pydeck as pdk
            layer = pdk.Layer("ScatterplotLayer", data=_df_nodes, get_position="[lon, lat]",
                              get_radius=60, get_fill_color="[220,20,60,220]", pickable=True)
            view = pdk.ViewState(latitude=clat, longitude=clon, zoom=12)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                     tooltip={"text": "{name}\n{desc}"}), use_container_width=True)
            rendered = True
        except Exception:
            pass

    if not rendered:
        fig = go.Figure(go.Scattermapbox(
            lat=_df_nodes["lat"], lon=_df_nodes["lon"], mode="markers",
            marker=dict(size=14, color="crimson"),
            text=_df_nodes["name"], hoverinfo="text",
        ))
        fig.update_layout(
            mapbox=dict(style="open-street-map", center=dict(lat=clat, lon=clon), zoom=12),
            height=480, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── TAB: Detections ─────────────────────────────────────────────
with tab_detect:
    st.markdown("#### Species detections")
    st.markdown('<p class="kn-muted">Heatmap of detections by species and hour. Filter by source, date, and minimum confidence.</p>', unsafe_allow_html=True)

    epoch = _get_epoch()

    # Load CSV metadata
    if drive_ok():
        bn_meta, kn_meta = list_csvs_drive_root(GDRIVE_FOLDER_ID, _epoch=epoch)
        bn_paths = [str(ensure_csv_cached(m, subdir="root/bn")) for m in bn_meta]
        kn_paths = [str(ensure_csv_cached(m, subdir="root/kn")) for m in kn_meta]
    else:
        bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
        bn_paths, kn_paths = list(bn_local), list(kn_local)

    bn_by_date = build_date_index(bn_paths, "bn") if bn_paths else {}
    kn_by_date = build_date_index(kn_paths, "kn") if kn_paths else {}

    # Controls
    with st.form(key=k("detect_form")):
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            src = st.selectbox("Source", ["KōreroNET", "BirdNET", "Combined"], index=0)
        with fc3:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01)

        if src == "Combined":
            day_options = sorted(set(bn_by_date) & set(kn_by_date))
        elif src == "BirdNET":
            day_options = sorted(bn_by_date.keys())
        else:
            day_options = sorted(kn_by_date.keys())

        with fc2:
            if day_options:
                d = st.date_input("Date", value=day_options[-1],
                                  min_value=day_options[0], max_value=day_options[-1])
            else:
                d = st.date_input("Date")

        submitted = st.form_submit_button("Update heatmap", use_container_width=True)

    if not day_options:
        st.warning("No dates available for the selected source. Try a different source or refresh data.")
    else:
        # Snap to nearest date
        day_set = set(day_options)
        if d not in day_set:
            earlier = [x for x in day_options if x <= d]
            d = earlier[-1] if earlier else day_options[0]
            st.caption(f"Date snapped to nearest available: **{d.isoformat()}**")

        def _load_filter(paths, kind, day):
            frames = []
            for p in (bn_by_date if kind == "bn" else kn_by_date).get(day, []):
                try:
                    raw = load_csv(p)
                    std = standardize_df(raw, kind)
                    std = std[std["ActualTime"].dt.date == day]
                    if not std.empty:
                        frames.append(std)
                except Exception:
                    pass
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        def _heatmap(df, min_c, title):
            df_f = df[pd.to_numeric(df["Confidence"], errors="coerce") >= min_c].copy()
            if df_f.empty:
                st.info("No detections above the confidence threshold for this date.")
                return
            df_f["Hour"] = df_f["ActualTime"].dt.hour
            hour_labels = {h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)}
            order = [hour_labels[h] for h in range(24)]
            df_f["HL"] = df_f["Hour"].map(hour_labels)
            pivot = df_f.groupby(["Label", "HL"]).size().unstack(fill_value=0).astype(int)
            for lbl in order:
                if lbl not in pivot.columns:
                    pivot[lbl] = 0
            pivot = pivot[order]
            totals = pivot.sum(axis=1)
            pivot = pivot.loc[totals.sort_values(ascending=False).index]

            fig = px.imshow(
                pivot.values, x=pivot.columns, y=pivot.index,
                color_continuous_scale="YlOrRd",
                labels=dict(x="Hour", y="Species", color="Count"),
                text_auto=True, aspect="auto",
            )
            fig.update_layout(
                title=dict(text=title, font=dict(size=16, family="DM Sans")),
                margin=dict(l=10, r=10, t=45, b=10),
                coloraxis_colorbar=dict(thickness=14, len=0.6),
                xaxis=dict(type="category"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Quick stats
            st.markdown(f"""
            <div class="kn-metrics">
                <div class="kn-metric"><div class="kn-metric-val">{int(totals.sum()):,}</div><div class="kn-metric-label">Total detections</div></div>
                <div class="kn-metric"><div class="kn-metric-val">{len(totals)}</div><div class="kn-metric-label">Species</div></div>
                <div class="kn-metric"><div class="kn-metric-val">{min_c:.0%}</div><div class="kn-metric-label">Confidence ≥</div></div>
            </div>
            """, unsafe_allow_html=True)

        with st.spinner("Building heatmap…"):
            kind_map = {"BirdNET": "bn", "KōreroNET": "kn"}
            if src == "Combined":
                df_all = pd.concat([_load_filter(bn_paths, "bn", d), _load_filter(kn_paths, "kn", d)], ignore_index=True)
                _heatmap(df_all, min_conf, f"Combined — {d.isoformat()}")
            else:
                kind_code = kind_map[src]
                df_single = _load_filter([], kind_code, d)
                _heatmap(df_single, min_conf, f"{src} — {d.isoformat()}")


# ── TAB: Verify ─────────────────────────────────────────────────
with tab_verify:
    st.markdown("#### Verify recordings")
    st.markdown('<p class="kn-muted">Browse and listen to audio chunks from backup snapshots. Select a species and navigate through detections.</p>', unsafe_allow_html=True)

    if not drive_ok():
        st.error("Google Drive is not configured. Add credentials in Streamlit secrets to enable verification.")
    else:
        with st.status("Indexing backup snapshots…", expanded=False) as status_v:
            try:
                master = build_master_index(GDRIVE_FOLDER_ID, _epoch=_get_epoch())
                status_v.update(label=f"Indexed {len(master):,} detections across {master['Date'].nunique() if not master.empty else 0} dates", state="complete")
            except Exception as exc:
                master = pd.DataFrame()
                status_v.update(label=f"Indexing failed: {exc!s}", state="error")

        if master.empty:
            st.info("No master CSVs found in backup snapshots.")
        else:
            # Filters
            fc1, fc2, fc3 = st.columns([2, 1, 2])
            with fc1:
                src_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], key=k("v_src"))
            with fc2:
                min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01, key=k("v_conf"))

            if src_v == "KōreroNET (KN)":
                pool = master[master["Kind"] == "KN"]
            elif src_v == "BirdNET (BN)":
                pool = master[master["Kind"] == "BN"]
            else:
                pool = master
            pool = pool[pd.to_numeric(pool["Confidence"], errors="coerce") >= min_conf_v]

            if pool.empty:
                st.info("No detections above the selected confidence threshold.")
            else:
                avail_days = sorted(pool["Date"].unique())
                with fc3:
                    day_v = st.date_input("Date", value=avail_days[-1],
                                          min_value=avail_days[0], max_value=avail_days[-1], key=k("v_day"))

                day_df = pool[pool["Date"] == day_v]
                if day_df.empty:
                    st.warning("No detections for the selected date.")
                else:
                    counts = day_df.groupby("Label").size().sort_values(ascending=False)
                    species_list = list(counts.index)

                    species = st.selectbox(
                        "Species", species_list,
                        format_func=lambda s: f"{s} — {counts[s]:,} detections",
                        key=k("v_species"),
                    )

                    playlist = (day_df[day_df["Label"] == species]
                                .sort_values(["Kind", "ChunkName"])
                                .reset_index(drop=True))

                    if playlist.empty:
                        st.info("No clips for this species.")
                    else:
                        # Navigation
                        idx_key = f"v_idx::{day_v}::{src_v}::{species}"
                        if idx_key not in st.session_state:
                            st.session_state[idx_key] = 0
                        idx = st.session_state[idx_key] % len(playlist)

                        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 6])
                        with c1:
                            if st.button("⏮ Prev", key=k("v_prev"), use_container_width=True):
                                idx = (idx - 1) % len(playlist)
                        with c2:
                            do_play = st.button("▶ Play", key=k("v_play"), type="primary", use_container_width=True)
                        with c3:
                            if st.button("⏭ Next", key=k("v_next"), use_container_width=True):
                                idx = (idx + 1) % len(playlist)
                        with c4:
                            st.markdown(f'<p class="kn-muted" style="padding-top:.6rem">{idx + 1} / {len(playlist)}</p>', unsafe_allow_html=True)

                        st.session_state[idx_key] = idx
                        row = playlist.iloc[idx]

                        # Metadata card
                        try:
                            conf_str = f"{float(row['Confidence']):.3f}"
                        except (ValueError, TypeError):
                            conf_str = "n/a"

                        st.markdown(f"""
                        <div class="kn-player">
                            <div class="kn-player-meta">
                                <strong>Chunk:</strong> {row['ChunkName']}<br>
                                <strong>Date:</strong> {row['Date']} &nbsp;·&nbsp;
                                <strong>Source:</strong> {row['Kind']} &nbsp;·&nbsp;
                                <strong>Confidence:</strong> {conf_str}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if do_play:
                            # Cooldown to avoid hammering audio playback
                            NAV_COOLDOWN = 0.4
                            now_ts = time.time()
                            last_play = st.session_state.get("_last_play_ts", 0.0)
                            if now_ts - last_play < NAV_COOLDOWN:
                                st.info("Just a moment between plays to keep the server happy.")
                            else:
                                st.session_state["_last_play_ts"] = now_ts
                                chunk_name = str(row.get("ChunkName", ""))
                                folder_id  = str(row.get("ChunkDriveFolderId", ""))
                                if not chunk_name or not folder_id:
                                    st.warning("No chunk mapping available for this detection.")
                                else:
                                    @_retry()
                                    def _safe_fetch(cn, fid, sd):
                                        return ensure_chunk_cached(cn, fid, subdir=sd, force=False)

                                    try:
                                        with st.spinner("Fetching audio…"):
                                            cached = _safe_fetch(chunk_name, folder_id, row["Kind"])
                                        if cached and cached.exists() and cached.stat().st_size > 0:
                                            if cached.stat().st_size > 30 * 1024 * 1024:
                                                st.warning("Audio file too large to preview (>30 MB).")
                                            else:
                                                try:
                                                    data = cached.read_bytes()
                                                    st.audio(data, format="audio/wav")
                                                except MemoryError:
                                                    st.warning("Not enough memory to load audio preview.")
                                        else:
                                            st.warning("Audio chunk not found in the Drive folder.")
                                    except Exception as exc:
                                        st.warning(f"Playback error: {exc!s}")


# ── TAB: Power ──────────────────────────────────────────────────
with tab_power:
    st.markdown("#### Node power history")
    st.markdown('<p class="kn-muted">State of Charge and energy consumption over the last 7 days. Data is stitched from hourly power logs.</p>', unsafe_allow_html=True)

    if not drive_ok():
        st.error("Google Drive is not configured.")
    else:
        logs_folder = find_subfolder(GDRIVE_FOLDER_ID, "Power logs")
        if not logs_folder:
            st.warning("No `Power logs` folder found under the Drive root.")
        else:
            if st.button("Load power data", type="primary", key=k("pwr_load"), use_container_width=False):
                with st.status("Building power time-series…", expanded=True) as pwr_status:
                    files = list_power_logs(logs_folder["id"], _epoch=_get_epoch())
                    cutoff = datetime.now() - timedelta(days=7)

                    frames: List[pd.DataFrame] = []
                    for meta in files:
                        local = ensure_log_cached(meta)
                        df = parse_power_log(local)
                        if df is None or df.empty:
                            continue
                        df = df[df["t"] >= cutoff]
                        if df.empty:
                            break
                        frames.append(df)

                    ts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["t", "PH_WH", "PH_mAh", "PH_SoCi", "PH_SoCv"])
                    pwr_status.update(label=f"Loaded {len(ts):,} data points", state="complete")

                if ts.empty:
                    st.info("No parsable power logs in the last 7 days.")
                else:
                    ts = ts.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_SoCi"], mode="lines",
                                            name="SoC (%)", line=dict(color="#22c55e", width=2.5), yaxis="y"))
                    fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_WH"], mode="lines",
                                            name="Energy (Wh)", line=dict(color="#f59e0b", width=2), yaxis="y2"))
                    fig.update_layout(
                        title=dict(text="Battery & Energy — last 7 days", font=dict(size=16, family="DM Sans")),
                        xaxis=dict(title=""),
                        yaxis=dict(title="State of Charge (%)", range=[0, 105], titlefont=dict(color="#22c55e")),
                        yaxis2=dict(title="Energy (Wh)", overlaying="y", side="right", titlefont=dict(color="#f59e0b")),
                        legend=dict(orientation="h", y=-0.12),
                        margin=dict(l=10, r=10, t=50, b=10),
                        hovermode="x unified",
                    )
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


# ── TAB: Log ────────────────────────────────────────────────────
with tab_log:
    st.markdown("#### GUI autostart log")
    st.markdown('<p class="kn-muted">Tail of the latest autostart log from the node. Newest lines first.</p>', unsafe_allow_html=True)

    if not drive_ok():
        st.error("Google Drive is not configured.")
    else:
        power_folder = find_subfolder(GDRIVE_FOLDER_ID, "Power logs")
        raw_folder = find_subfolder(power_folder["id"], "raw") if power_folder else None

        if not raw_folder:
            st.warning("Could not locate `Power logs/raw` in Drive.")
        else:
            if st.button("Load latest log", type="primary", key=k("log_load")):
                files = list_autostart_logs(raw_folder["id"], _epoch=_get_epoch())
                if not files:
                    st.info("No `*__gui_autostart.log` files found.")
                else:
                    latest = files[0]
                    local = ensure_log_cached(latest, force=True)

                    st.caption(f"File: `{latest['name']}` — {local.stat().st_size:,} bytes")

                    # Tail last 500 lines
                    def _tail(path: Path, n: int = 500) -> List[str]:
                        try:
                            if path.stat().st_size == 0:
                                return ["(empty file)"]
                            with open(path, "rb") as f:
                                f.seek(0, 2)
                                end = f.tell()
                                buf, lines = b"", []
                                while end > 0 and len(lines) <= n:
                                    chunk_size = min(8192, end)
                                    end -= chunk_size
                                    f.seek(end)
                                    buf = f.read(chunk_size) + buf
                                    lines = buf.split(b"\n")
                            return buf.decode("utf-8", errors="replace").splitlines()[-n:]
                        except Exception as exc:
                            return [f"[error] {exc!s}"]

                    lines = list(reversed(_tail(local, 500)))
                    numbered = "\n".join(f"{i + 1:>4}  {line}" for i, line in enumerate(lines))
                    st.code(numbered, language="log")

                    st.download_button("Download full log", local.read_bytes(),
                                       file_name=latest["name"], mime="text/plain", key=k("dl_log"))
