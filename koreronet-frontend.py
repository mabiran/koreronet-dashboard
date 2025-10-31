#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KōreroNET Dashboard (with splash + Drive)
# ------------------------------------------------------------
# - Tab 1: Root CSV heatmaps (Drive or local) with calendar + min confidence
#          NEW: Click a heatmap cell → mini player (Prev / Play / Next) for
#               matching clips when using new master CSVs (≥ 31 Oct 2025).
#          NEW: "Load" flow + progress bar; changing controls hides player
#               until Load is pressed (prevents stale/cached views).
# - Tab 2: Verify by snapshot date, uses master CSVs and known chunk folders.
# - Tab 3: Power graph from "Power logs" (Drive).
#
# Drive freshness:
#   - Auto: if we detect newer modifiedTime in root/Backup/, we clear caches.
#   - Manual: "Refresh from Drive now" button also clears caches.
#
# Requires (for Tab 1 click handling):
#   streamlit-plotly-events
#
# Streamlit secrets required:
#   GDRIVE_FOLDER_ID = "your_root_folder_id"
#   [service_account] ... (standard service-account JSON fields)
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

# Optional click support
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

# ─────────────────────────────────────────────────────────────
# Page style
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="KōreroNET Dashboard", layout="wide")
# ─────────────────────────────────────────────────────────────
# Splash / Gate (no hard lock)
# ─────────────────────────────────────────────────────────────
def show_overlay():
    st.markdown(
        """
        <style>
        .overlay {
          position: fixed; inset: 0; z-index: 9999;
          display: grid; place-items: center;
          background: radial-gradient(1000px 500px at 50% -10%, #1a1a1a 0%, #0b0b0b 60%, #070707 100%);
        }
        .overlay-inner { text-align:center; padding: 2rem; }
        .brand-title {font-size: clamp(48px, 8vw, 96px); font-weight: 800; letter-spacing:.02em;}
        .brand-sub {font-size: clamp(28px, 4vw, 48px); font-weight:600; opacity:.9; margin-top:.4rem;}
        </style>
        <div class="overlay">
          <div class="overlay-inner">
            <div class="brand-title">KōreroNET</div>
            <div class="brand-sub">AUT • GeoEnviroSense</div>
            <p style="opacity:.85;margin:.75rem 0 1.25rem">initialising…</p>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([2,2,2])
    with c2:
        if st.button("Enter dashboard", type="primary", use_container_width=True, key="enter_btn"):
            st.session_state["entered"] = True
            st.rerun()
    st.markdown(
        '<p style="text-align:center;margin-top:.75rem;opacity:.7">Having trouble? '
        'Append <code>?skip_splash=1</code> to the URL.</p></div></div>',
        unsafe_allow_html=True,
    )

# Bypass via URL: …/app?skip_splash=1
skip = st.query_params.get("skip_splash", ["0"])[0] if hasattr(st, "query_params") else "0"
if skip == "1":
    st.session_state["entered"] = True

if not st.session_state.get("entered", False):
    show_overlay()
    # NOTE: No st.stop() — UI renders underneath but overlay blocks until clicked.


# ============================================================================
# Constants & Regex
# ============================================================================
# chunk filename pattern (legacy-style chunks)
CHUNK_RE = re.compile(
    r"^(?P<root>\d{8}_\d{6})__(?P<tag>bn|kn)_(?P<s>\d+\.\d{2})_(?P<e>\d+\.\d{2})__(?P<label>.+?)__p(?P<conf>\d+\.\d{2})\.wav$",
    re.IGNORECASE,
)
# new root CSVs in Drive root (≥ 31 Oct)
NEW_ROOT_BN = re.compile(r"^(\d{8}_\d{6})_birdnet_master\.csv$", re.IGNORECASE)
NEW_ROOT_KN = re.compile(r"^(\d{8}_\d{6})_koreronet_master\.csv$", re.IGNORECASE)
# snapshot folder name
SNAP_RE     = re.compile(r"^(\d{8})_(\d{6})$", re.IGNORECASE)
# format cutoff
CUTOFF_NEW  = date(2025, 10, 31)

# ============================================================================
# Splash once
# ============================================================================
if "splash_done" not in st.session_state:
    st.session_state["splash_done"] = True
    st.markdown(
        """
        <div class="center-wrap fade-enter">
          <div>
            <div class="brand-title">KōreroNET</div>
            <div class="brand-sub">AUT</div>
            <div class="small" style="margin-top:10px;">initialising…</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

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

# Top controls + manual refresh
row_top = st.columns([3,1])
with row_top[0]:
    node = st.selectbox("Node Select", ["Auckland-Orākei"], index=0)
with row_top[1]:
    if st.button("🔄 Refresh from Drive now"):
        st.cache_data.clear()
        # nuke session maps that depend on epoch
        for k in list(st.session_state.keys()):
            if str(k).startswith(("drive_kids::","DRIVE_EPOCH","root_map::","t1_idx::","t1_loaded_for::")):
                del st.session_state[k]
        st.success("Cache cleared.")
        st.rerun()

# ============================================================================
# Secrets / Drive
# ============================================================================
GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID", None)

def _normalize_private_key(pk: str) -> str:
    if not isinstance(pk, str): return pk
    return pk.replace("\\n", "\n") if "\\n" in pk else pk

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

# ============================================================================
# Drive helpers + epoch (freshness)
# ============================================================================
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    drive = get_drive_client()
    if not drive: return []
    items, token = [], None
    while True:
        page_size = min(100, max_items - len(items))
        if page_size <= 0: break
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,md5Checksum,parents)",
            pageSize=page_size,
            pageToken=token,
            orderBy="folder,name_natural",
            includeItemsFromAllDrives=True, supportsAllDrives=True, corpora="allDrives",
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token: break
    return items

@st.cache_data(ttl=180, show_spinner=False)
def _compute_drive_epoch(root_id: str) -> str:
    drive = get_drive_client()
    if not drive: return "no-drive"
    def _max_mtime(kids: List[Dict[str, Any]]) -> str:
        if not kids: return ""
        return max((k.get("modifiedTime","") for k in kids), default="")
    root_kids = list_children(root_id, max_items=2000)
    root_max  = _max_mtime(root_kids)
    # include Backup one level deep
    bkid = next((k["id"] for k in root_kids
                 if k.get("mimeType")=="application/vnd.google-apps.folder"
                 and k.get("name","").lower()=="backup"), None)
    back_max = ""
    if bkid:
        back_kids = list_children(bkid, max_items=2000)
        back_max  = _max_mtime(back_kids)
    token = "|".join([root_max, back_max])
    return token or time.strftime("%Y%m%d%H%M%S")

def _ensure_epoch_key():
    if not drive_enabled(): return
    new_epoch = _compute_drive_epoch(GDRIVE_FOLDER_ID)
    old_epoch = st.session_state.get("DRIVE_EPOCH")
    if old_epoch is None:
        st.session_state["DRIVE_EPOCH"] = new_epoch
    elif new_epoch != old_epoch:
        st.cache_data.clear()
        # reset maps dependent on epoch
        for k in list(st.session_state.keys()):
            if str(k).startswith(("drive_kids::","root_map::","t1_idx::","t1_loaded_for::")):
                del st.session_state[k]
        st.session_state["DRIVE_EPOCH"] = new_epoch

_ensure_epoch_key()

def _folder_children_cached(folder_id: str) -> List[Dict[str, Any]]:
    epoch = st.session_state.get("DRIVE_EPOCH", "0")
    key = f"drive_kids::{epoch}::{folder_id}"
    if key not in st.session_state:
        st.session_state[key] = list_children(folder_id, max_items=2000)
    return st.session_state[key]

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

def ensure_csv_cached(meta: Dict[str, Any], subdir: str, cache_epoch: str = "") -> Path:
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

    kids = _folder_children_cached(folder_id)
    name_to_id = {k.get("name"): k.get("id") for k in kids}
    if chunk_name in name_to_id:
        try:
            download_to(local_path, name_to_id[chunk_name])
            return local_path
        except Exception:
            return None

    m = CHUNK_RE.match(chunk_name or "")
    if not m: return None
    target = {
        "root": m.group("root"), "tag": m.group("tag").lower(),
        "s": float(m.group("s")), "e": float(m.group("e")),
        "label": m.group("label").lower()
    }
    cands = []
    for k in kids:
        nm = k.get("name","")
        if not nm.lower().endswith(".wav"): continue
        mm = CHUNK_RE.match(nm)
        if not mm: continue
        info = {
            "root": mm.group("root"), "tag": mm.group("tag").lower(),
            "s": float(mm.group("s")), "e": float(mm.group("e")),
            "label": mm.group("label").lower()
        }
        if info["root"]==target["root"] and info["tag"]==target["tag"] and info["label"]==target["label"]:
            cands.append((info, k.get("id"), nm))
    if not cands: return None
    tol = 0.75
    def score(info):
        s_c, e_c = info["s"], info["e"]; s_t, e_t = target["s"], target["e"]
        contains = (s_c - tol) <= s_t and (e_c + tol) >= e_t
        cen_diff = abs(((s_c + e_c)/2) - ((s_t + e_t)/2))
        return (0 if contains else 1, cen_diff)
    cands.sort(key=lambda it: score(it[0]))
    best = cands[0]
    try:
        download_to(local_path, best[1])
        st.caption(f"⚠️ Fuzzy match: requested `{chunk_name}` → `{best[2]}`")
        return local_path
    except Exception:
        return None

# ============================================================================
# Root CSV listing (Tab 1)
# ============================================================================
@st.cache_data(show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    # also new-format if saved locally
    bn_paths += sorted(glob.glob(os.path.join(root, "[0-9]"*8 + "_" + "[0-9]"*6 + "_birdnet_master.csv")))
    kn_paths += sorted(glob.glob(os.path.join(root, "[0-9]"*8 + "_" + "[0-9]"*6 + "_koreronet_master.csv")))
    return bn_paths, kn_paths

@st.cache_data(show_spinner=False)
def list_csvs_drive_root(folder_id: str, cache_epoch: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kids = list_children(folder_id, max_items=2000)
    bn, kn = [], []
    for k in kids:
        n = k.get("name",""); nl = n.lower()
        if nl.endswith(".csv"):
            if nl.startswith("bn") or NEW_ROOT_BN.match(n):
                bn.append(k)
            elif nl.startswith("kn") or NEW_ROOT_KN.match(n):
                kn.append(k)
    bn.sort(key=lambda m: m.get("name","")); kn.sort(key=lambda m: m.get("name",""))
    return bn, kn

def _std_newformat(df: pd.DataFrame) -> pd.DataFrame:
    # expects Clip, ActualStartTime, Label, Probability
    out = pd.DataFrame()
    out["Label"]       = df.get("Label", "Unknown")
    out["Confidence"]  = pd.to_numeric(df.get("Probability", np.nan), errors="coerce")
    out["ActualTime"]  = pd.to_datetime(df.get("ActualStartTime", pd.NaT), errors="coerce")
    out["Clip"]        = df.get("Clip", np.nan)
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
    df["Clip"] = np.nan  # legacy root CSVs lack direct clip mapping
    return df[["Label", "Confidence", "ActualTime", "Clip"]]

def standardize_root_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    cols = set(c.lower() for c in df.columns)
    if {"clip","actualstarttime","label","probability"} <= cols:
        return _std_newformat(df)
    return _std_legacy(df, kind)

def _find_backup_folder(root_folder_id: str) -> Optional[Dict[str, Any]]:
    kids = list_children(root_folder_id, max_items=2000)
    return next((k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder"
                 and k.get("name","").lower()=="backup"), None)

def _find_chunk_dirs(snapshot_id: str) -> Dict[str, str]:
    kids = list_children(snapshot_id, max_items=2000)
    kn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "koreronet" in k.get("name","").lower()]
    bn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "birdnet"   in k.get("name","").lower()]
    return {"KN": (kn[0]["id"] if kn else snapshot_id), "BN": (bn[0]["id"] if bn else snapshot_id)}

@st.cache_data(show_spinner=False)
def map_root_csvname_to_chunk_folder(root_folder_id: str, csv_name: str) -> Optional[str]:
    """
    For new-format root CSVs, find Backup/<timestamp>/<koreronet|birdnet> folder id.
    Returns folder_id or None.
    """
    m_kn = NEW_ROOT_KN.match(csv_name or "")
    m_bn = NEW_ROOT_BN.match(csv_name or "")
    if not (m_kn or m_bn):
        return None
    ts = (m_kn or m_bn).group(1)  # YYYYMMDD_HHMMSS
    backup = _find_backup_folder(root_folder_id)
    if not backup: return None
    snaps = list_children(backup["id"], max_items=2000)
    snap = next((s for s in snaps if s.get("mimeType")=="application/vnd.google-apps.folder" and s.get("name","")==ts), None)
    if not snap: return None
    dirs = _find_chunk_dirs(snap["id"])
    return dirs["BN"] if m_bn else dirs["KN"]

# ============================================================================
# Snapshot/master logic (Tab 2)
# ============================================================================
def _parse_date_from_snapname(name: str) -> Optional[date]:
    m = SNAP_RE.match(name or "")
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None

def _find_named_in_snapshot(snapshot_id: str, exact_name: str) -> Optional[Dict[str, Any]]:
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

def _compose_chunk_name(kind: str, wav_base: str, start: float, end: float, label: str, conf: float) -> str:
    root = Path(wav_base).stem
    start_s = f"{float(start):06.2f}"
    end_s   = f"{float(end):06.2f}"
    lab     = re.sub(r"[^A-Za-z0-9]+", "_", str(label or "")).strip("_")
    pc      = f"{float(conf):.2f}"
    tag     = "bn" if kind.lower()=="bn" else "kn"
    return f"{root}__{tag}_{start_s}_{end_s}__{lab}__p{pc}.wav"

@st.cache_data(show_spinner=True)
def build_master_index_by_snapshot_date(root_folder_id: str, cache_epoch: str) -> pd.DataFrame:
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
        dirs = _find_chunk_dirs(snap_id)

        if snap_date >= CUTOFF_NEW:
            # koreronet_master
            kn_meta = _find_named_in_snapshot(snap_id, "koreronet_master.csv")
            if kn_meta:
                local = ensure_csv_cached(kn_meta, subdir=f"snap_{snap_id}/koreronet", cache_epoch=cache_epoch)
                try:
                    df = pd.read_csv(local)
                    for _, r in df.iterrows():
                        rows.append({
                            "Date": snap_date, "Kind": "KN", "Label": str(r.get("Label","Unknown")),
                            "Confidence": float(r.get("Probability", np.nan)),
                            "Start": np.nan, "End": np.nan, "WavBase": "",
                            "ChunkName": str(r.get("Clip","")).strip(),
                            "ChunkDriveFolderId": dirs["KN"], "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass
            # birdnet_master
            bn_meta = _find_named_in_snapshot(snap_id, "birdnet_master.csv")
            if bn_meta:
                local = ensure_csv_cached(bn_meta, subdir=f"snap_{snap_id}/birdnet", cache_epoch=cache_epoch)
                try:
                    df = pd.read_csv(local)
                    for _, r in df.iterrows():
                        rows.append({
                            "Date": snap_date, "Kind": "BN", "Label": str(r.get("Label", r.get("Common name","Unknown"))),
                            "Confidence": float(r.get("Probability", r.get("Confidence", np.nan))),
                            "Start": np.nan, "End": np.nan, "WavBase": "",
                            "ChunkName": str(r.get("Clip","")).strip(),
                            "ChunkDriveFolderId": dirs["BN"], "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass
        else:
            # Legacy (synthesize chunk name from detect CSVs)
            root_kids = list_children(snap_id, max_items=2000)
            files_only = [f for f in root_kids if f.get("mimeType") != "application/vnd.google-apps.folder"]

            def _match_legacy_name(name: str, kind: str) -> bool:
                n = name.lower()
                if kind == "KN":
                    return (("koreronet" in n) and ("detect" in n) and n.endswith(".csv"))
                else:
                    return (("birdnet" in n) and ("detect" in n) and n.endswith(".csv"))

            # KN
            kn_meta = next((f for f in files_only if _match_legacy_name(f.get("name",""), "KN")), None)
            if not kn_meta:
                for sf in [f for f in root_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]:
                    cand = [f for f in list_children(sf["id"], max_items=2000)
                            if f.get("mimeType") != "application/vnd.google-apps.folder"
                            and _match_legacy_name(f.get("name",""), "KN")]
                    if cand: kn_meta = cand[0]; break
            if kn_meta:
                local = ensure_csv_cached(kn_meta, subdir=f"snap_{snap_id}/koreronet", cache_epoch=cache_epoch)
                try:
                    df = pd.read_csv(local)
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
                            "ChunkDriveFolderId": dirs["KN"], "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass
            # BN
            bn_meta = next((f for f in files_only if _match_legacy_name(f.get("name",""), "BN")), None)
            if not bn_meta:
                for sf in [f for f in root_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]:
                    cand = [f for f in list_children(sf["id"], max_items=2000)
                            if f.get("mimeType") != "application/vnd.google-apps.folder"
                            and _match_legacy_name(f.get("name",""), "BN")]
                    if cand: bn_meta = cand[0]; break
            if bn_meta:
                local = ensure_csv_cached(bn_meta, subdir=f"snap_{snap_id}/birdnet", cache_epoch=cache_epoch)
                try:
                    df = pd.read_csv(local)
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
                            "ChunkDriveFolderId": dirs["BN"], "SnapId": snap_id, "SnapName": snap_name,
                        })
                except Exception:
                    pass

    if not rows:
        return pd.DataFrame(columns=[
            "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
            "ChunkDriveFolderId","SnapId","SnapName"
        ])
    out = pd.DataFrame(rows).sort_values(["Date","Kind","Label"], ascending=[False,True,True]).reset_index(drop=True)
    return out

# ============================================================================
# Tabs
# ============================================================================
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "⚡ Power"])

# =========================
# TAB 1 — Detections (root) with Load + click-to-play
# =========================
with tab1:
    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
    # resolve root files
    if drive_enabled():
        bn_meta, kn_meta = list_csvs_drive_root(GDRIVE_FOLDER_ID, cache_epoch=cache_epoch)
        root_map: Dict[str, Optional[str]] = {}
        bn_paths, kn_paths = [], []
        # download + map each new-format root CSV to its chunk folder id
        for m in bn_meta:
            local = ensure_csv_cached(m, subdir="root/bn", cache_epoch=cache_epoch)
            bn_paths.append(local)
            root_map[str(local)] = map_root_csvname_to_chunk_folder(GDRIVE_FOLDER_ID, m["name"])
        for m in kn_meta:
            local = ensure_csv_cached(m, subdir="root/kn", cache_epoch=cache_epoch)
            kn_paths.append(local)
            root_map[str(local)] = map_root_csvname_to_chunk_folder(GDRIVE_FOLDER_ID, m["name"])
        st.session_state[f"root_map::{cache_epoch}"] = root_map
    else:
        bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
        bn_paths = [Path(p) for p in bn_local]
        kn_paths = [Path(p) for p in kn_local]
        st.session_state[f"root_map::{cache_epoch}"] = {}

    # build date index
    @st.cache_data(show_spinner=False)
    def extract_dates_from_csv(path: str | Path) -> List[date]:
        path = str(path)
        dates = set()
        try:
            for chunk in pd.read_csv(path, chunksize=5000):
                if "ActualStartTime" in chunk.columns:
                    s = pd.to_datetime(chunk["ActualStartTime"], errors="coerce")
                else:
                    s = pd.to_datetime(chunk.get("ActualTime", pd.NaT), errors="coerce", dayfirst=True)
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

    bn_by_date = build_date_index(bn_paths, "bn") if bn_paths else {}
    kn_by_date = build_date_index(kn_paths, "kn") if kn_paths else {}

    # controls
    c1, c2 = st.columns([1.5, 1])
    with c1:
        src = st.selectbox("Source", ["KōreroNET (kn)", "BirdNET (bn)", "Combined"], index=0, key="t1_src")
    with c2:
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01, key="t1_conf")

    if src == "Combined":
        options, help_txt = sorted(set(bn_by_date.keys()).intersection(set(kn_by_date.keys()))), "Dates with BOTH BN & KN."
    elif src == "BirdNET (bn)":
        options, help_txt = sorted(bn_by_date.keys()), "Dates present in any BN file."
    else:
        options, help_txt = sorted(kn_by_date.keys()), "Dates present in any KN file."

    if not options:
        st.warning(f"No available dates for {src}."); st.stop()

    # date input
    d = st.date_input("Day", value=options[-1], min_value=options[0], max_value=options[-1], help=help_txt, key="t1_day")

    # Mark dirty whenever controls change
    controls_signature = (src, float(min_conf), d.isoformat())
    if st.session_state.get("t1_controls_sig") != controls_signature:
        st.session_state["t1_controls_sig"] = controls_signature
        st.session_state["t1_loaded_for"] = None  # force hide player/panel until Load

    # Load button + progress bar
    if st.button("Load detections", type="primary", key="tab1_load_btn"):
        st.session_state["t1_loaded_for"] = controls_signature

    if st.session_state.get("t1_loaded_for") != controls_signature:
        st.info("Adjust settings and click **Load detections**.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def load_csv(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(str(path))

    def load_and_filter_with_progress(paths: List[str], kind: str, day_selected: date) -> pd.DataFrame:
        frames = []
        root_map = st.session_state.get(f"root_map::{cache_epoch}", {})
        if not paths:
            return pd.DataFrame(columns=["Label","Confidence","ActualTime","Clip","Kind","ChunkDriveFolderId"])
        pb = st.progress(0, text="Loading…")
        for i, p in enumerate(paths, 1):
            try:
                raw = load_csv(p)
                std = standardize_root_df(raw, kind)
                std = std[std["ActualTime"].dt.date == day_selected]
                if not std.empty:
                    std["Kind"] = "BN" if kind == "bn" else "KN"
                    std["ChunkDriveFolderId"] = root_map.get(str(p), None)  # only present for new format
                    frames.append(std)
            except Exception:
                pass
            pb.progress(int(i/len(paths)*100), text=f"Loading… ({i}/{len(paths)})")
        pb.empty()
        if not frames:
            return pd.DataFrame(columns=["Label","Confidence","ActualTime","Clip","Kind","ChunkDriveFolderId"])
        return pd.concat(frames, ignore_index=True)

    # gather file list for the chosen date
    def files_for_day(by_date_map: Dict[date, List[str]], sel_day: date) -> List[str]:
        return by_date_map.get(sel_day, [])

    if src == "BirdNET (bn)":
        paths = files_for_day(bn_by_date, d)
        df_show = load_and_filter_with_progress(paths, "bn", d)
        title = f"BirdNET • {d.isoformat()}"
    elif src == "KōreroNET (kn)":
        paths = files_for_day(kn_by_date, d)
        df_show = load_and_filter_with_progress(paths, "kn", d)
        title = f"KōreroNET • {d.isoformat()}"
    else:
        paths_bn = files_for_day(bn_by_date, d)
        paths_kn = files_for_day(kn_by_date, d)
        df_bn = load_and_filter_with_progress(paths_bn, "bn", d)
        df_kn = load_and_filter_with_progress(paths_kn, "kn", d)
        df_show = pd.concat([df_bn, df_kn], ignore_index=True)
        title = f"Combined (BN+KN) • {d.isoformat()}"

    # apply confidence filter for heatmap but keep a copy for click panel
    df_all = df_show.copy()
    df_plot = df_show[pd.to_numeric(df_show["Confidence"], errors="coerce") >= float(min_conf)].copy()
    if df_plot.empty:
        st.warning("No detections after filtering.")
        st.stop()

    # Prepare heatmap
    df_plot["Hour"] = df_plot["ActualTime"].dt.hour
    hour_labels = {h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)}
    order = [hour_labels[h] for h in range(24)]
    df_plot["HourLabel"] = df_plot["Hour"].map(hour_labels)

    pivot = df_plot.groupby(["Label","HourLabel"]).size().unstack(fill_value=0).astype(int)
    for lbl in order:
        if lbl not in pivot.columns: pivot[lbl] = 0
    pivot = pivot[order]
    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values(ascending=False).index]

    fig = px.imshow(
        pivot.values,
        x=pivot.columns, y=pivot.index,
        color_continuous_scale="RdYlBu_r",
        labels=dict(x="Hour (AM/PM)", y="Species (label)", color="Detections"),
        text_auto=True, aspect="auto", title=title,
    )
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    fig.update_xaxes(type="category")

    # Show chart + click handling
    if plotly_events is None:
        st.plotly_chart(fig, use_container_width=True)
        st.info("Tip: install `streamlit-plotly-events` to enable click-to-play on this heatmap.")
        st.stop()

    clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                           override_height=480, override_width="100%")
    if not clicks:
        st.stop()

    # One click → x = HourLabel, y = Label
    x = clicks[0].get("x")
    y = clicks[0].get("y")
    if x is None or y is None:
        st.stop()

    # Build HourLabel in the unfiltered copy (so we can include rows < min_conf if needed)
    df_all["Hour"] = df_all["ActualTime"].dt.hour
    df_all["HourLabel"] = df_all["Hour"].map(hour_labels)

    cell_rows = df_all[(df_all["Label"]==y) & (df_all["HourLabel"]==x)].copy()
    playable = cell_rows.dropna(subset=["Clip","ChunkDriveFolderId"])

    st.markdown(f"**Selected:** {y} @ {x} — {len(cell_rows)} detections ({len(playable)} playable)")

    if playable.empty:
        st.info("These detections are from legacy root CSVs (no direct clip paths). Use **Tab 2** to verify by snapshot.")
        st.stop()

    # Playlist controls (per cell)
    key_prefix = f"t1_idx::{d.isoformat()}::{title}::{y}::{x}"
    if key_prefix not in st.session_state: st.session_state[key_prefix] = 0
    idx = st.session_state[key_prefix] % len(playable)

    c1, c2, c3, _ = st.columns([1,1,1,6])
    autoplay = False
    with c1:
        if st.button("⏮ Prev", key=key_prefix+"_prev"): idx = (idx-1) % len(playable); autoplay = True
    with c2:
        if st.button("▶ Play", key=key_prefix+"_play"): autoplay = True
    with c3:
        if st.button("⏭ Next", key=key_prefix+"_next"): idx = (idx+1) % len(playable); autoplay = True
    st.session_state[key_prefix] = idx

    row = playable.iloc[idx]
    chunk = str(row.get("Clip",""))
    folder_id = str(row.get("ChunkDriveFolderId",""))
    kind = str(row.get("Kind","KN"))
    st.caption(f"Chunk: `{chunk}` • Kind: {kind}")

    with st.spinner("Fetching audio…"):
        cached = ensure_chunk_cached(chunk, folder_id, subdir=kind)
    if not cached or not cached.exists():
        st.warning("Audio chunk not found in Drive folder.")
        st.stop()
    try:
        with open(cached, "rb") as f:
            data = f.read()
    except Exception:
        st.warning("Cannot open audio chunk.")
        st.stop()

    if autoplay:
        import base64
        b64 = base64.b64encode(data).decode()
        st.markdown(f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>', unsafe_allow_html=True)
    else:
        st.audio(data, format="audio/wav")

# ================================
# TAB 2 — Verify (snapshot date)
# ================================
with tab_verify:
    if not drive_enabled():
        st.error("Google Drive is not configured in secrets."); st.stop()

    st.markdown("##### Indexing master CSVs by snapshot date…")
    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
    master = build_master_index_by_snapshot_date(GDRIVE_FOLDER_ID, cache_epoch=cache_epoch)

    if master.empty:
        st.warning("No master CSVs found in any snapshot.")
        st.stop()

    colA, colB = st.columns([2,1])
    with colA:
        src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0)
    with colB:
        min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01)

    if src_mode_v == "KōreroNET (KN)":
        pool = master[master["Kind"]=="KN"]
    elif src_mode_v == "BirdNET (BN)":
        pool = master[master["Kind"]=="BN"]
    else:
        pool = master
    pool = pool[pd.to_numeric(pool["Confidence"], errors="coerce") >= float(min_conf_v)]
    if pool.empty:
        st.info("No rows above the selected confidence."); st.stop()

    avail_days = sorted(pool["Date"].unique())
    day_pick = st.date_input("Day", value=avail_days[-1], min_value=avail_days[0], max_value=avail_days[-1])

    day_df = pool[pool["Date"] == day_pick]
    if day_df.empty:
        st.warning("No detections for the chosen date."); st.stop()

    counts = day_df.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox("Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_pick.isoformat()}::{src_mode_v}",
    )

    playlist = day_df[day_df["Label"] == species].sort_values(["Kind","ChunkName"]).reset_index(drop=True)

    idx_key = f"v2_idx::{day_pick.isoformat()}::{src_mode_v}::{species}"
    if idx_key not in st.session_state: st.session_state[idx_key] = 0
    idx = st.session_state[idx_key] % len(playlist)

    col1, col2, col3, col4 = st.columns([1,1,1,6])
    autoplay = False
    with col1:
        if st.button("⏮ Prev"): idx = (idx - 1) % len(playlist); autoplay = True
    with col2:
        if st.button("▶ Play"): autoplay = True
    with col3:
        if st.button("⏭ Next"): idx = (idx + 1) % len(playlist); autoplay = True
    st.session_state[idx_key] = idx

    row = playlist.iloc[idx]
    st.markdown(f"**Date:** {row['Date']}  |  **Chunk:** `{row['ChunkName']}`  |  **Kind:** {row['Kind']}  |  **Confidence:** {float(row['Confidence']):.3f}")

    def _play_audio(row_: pd.Series, auto: bool):
        chunk_name = str(row_.get("ChunkName","") or "")
        folder_id  = str(row_.get("ChunkDriveFolderId","") or "")
        kind       = str(row_.get("Kind","UNK"))
        if not (chunk_name and folder_id):
            st.warning("No chunk mapping available."); return
        with st.spinner("Fetching audio…"):
            cached = ensure_chunk_cached(chunk_name, folder_id, subdir=kind)
        if not cached or not cached.exists():
            st.warning("Audio chunk not found in Drive folder."); return
        try:
            with open(cached, "rb") as f:
                data = f.read()
        except Exception:
            st.warning("Cannot open audio chunk."); return
        if auto:
            import base64
            b64 = base64.b64encode(data).decode()
            st.markdown(f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>', unsafe_allow_html=True)
        else:
            st.audio(data, format="audio/wav")

    _play_audio(row, autoplay)

# =====================
# TAB 3 — Power graph
# =====================
with tab3:
    st.subheader("Node Power History")
    if not drive_enabled():
        st.error("Google Drive is not configured in secrets."); st.stop()

    def find_subfolder_by_name(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
        kids = list_children(root_id, max_items=2000)
        return next((k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder"
                     and k.get("name","").lower()==name_ci.lower()), None)

    logs_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
    if not logs_folder:
        st.warning("Could not find 'Power logs' folder under the Drive root.")
        st.stop()

    LOG_RE = re.compile(r"^power_history_(\d{8})_(\d{6})\.log$", re.IGNORECASE)

    @st.cache_data(show_spinner=True)
    def list_power_logs(folder_id: str, cache_epoch: str) -> List[Dict[str, Any]]:
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
        try: payload = line.split(",", 1)[1]
        except Exception: return []
        vals = []
        for tok in payload.strip().split(","):
            tok = tok.strip().rstrip(".")
            try: vals.append(float(tok))
            except Exception: pass
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
                m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", l)
                if m:
                    try:
                        head_dt = datetime.strptime(m.group(0), "%Y-%m-%d %H:%M:%S")
                        break
                    except Exception:
                        pass
        if head_dt is None: return None

        wh_line   = next((l for l in lines if l.upper().startswith("PH_WH")),   "")
        mah_line  = next((l for l in lines if l.upper().startswith("PH_MAH")),  "")
        soci_line = next((l for l in lines if l.upper().startswith("PH_SOCI")), "")
        socv_line = next((l for l in lines if l.upper().startswith("PH_SOCV")), "")

        def pad(arr, n): return [np.nan]*(n-len(arr)) + arr if len(arr) < n else arr[:n]

        WH, mAh = _parse_float_list(wh_line), _parse_float_list(mah_line)
        SoCi, SoCv = _parse_float_list(soci_line), _parse_float_list(socv_line)
        L = max(len(WH), len(mAh), len(SoCi), len(SoCv))
        if L == 0: return None

        WH, mAh, SoCi, SoCv = pad(WH, L), pad(mAh, L), pad(SoCi, L), pad(SoCv, L)
        times = [head_dt - timedelta(hours=(L-1 - i)) for i in range(L)]
        return pd.DataFrame({"t": times, "PH_WH": WH, "PH_mAh": mAh, "PH_SoCi": SoCi, "PH_SoCv": SoCv})

    cache_epoch = st.session_state.get("DRIVE_EPOCH", "0")
    if st.button("Show results", type="primary", key="tab3_show_btn"):
        with st.spinner("Building power time-series…"):
            files = list_power_logs(logs_folder["id"], cache_epoch=cache_epoch)
            frames: List[pd.DataFrame] = []
            for meta in files[:2]:
                local = ensure_log_cached(meta)
                df = parse_power_log(local)
                if df is not None and not df.empty:
                    frames.append(df)
            ts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["t","PH_WH","PH_mAh","PH_SoCi","PH_SoCv"])

        if ts.empty:
            st.warning("No parsable power logs found.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_SoCi"], mode="lines", name="SoC_i (%)", yaxis="y1"))
            fig.add_trace(go.Scatter(x=ts["t"], y=ts["PH_WH"],  mode="lines", name="Energy (Wh)", yaxis="y2"))
            fig.update_layout(
                title="Power / State of Charge over time",
                xaxis=dict(title="Time"),
                yaxis=dict(title="SoC (%)", range=[0,100]),
                yaxis2=dict(title="Wh", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            last = ts.iloc[-1]
            c1, c2 = st.columns(2)
            with c1: st.metric("Last SoC_i (%)", f"{last['PH_SoCi']:.1f}")
            with c2: st.metric("Last Energy (Wh)", f"{last['PH_WH']:.2f}")
