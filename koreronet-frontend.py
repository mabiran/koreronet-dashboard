#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KōreroNET Dashboard — Google Drive (masters inside snapshot root or subfolders)

• Tab 1 (Detections): calendar input limited by min/max, snaps to nearest available day if empty.
• Tab 2 (Verify): uses master CSVs per snapshot (no per-file crawling), calendar input with the same snapping.
• Tab 3 (Drive): quick browser.

Drive lookup:
- Finds master CSVs flexibly anywhere under each snapshot (root OR one-level subfolders).
- Finds chunk folders named like "koreronet" / "birdnet"; falls back to snapshot root if not found.
- Downloads audio chunks on demand only.
"""

import os, io, re, glob, json
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Page style
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="KōreroNET Dashboard", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:1rem;}
.center-wrap {display:flex; align-items:center; justify-content:center; min-height:140px;}
.fade {opacity:0.95; transition: opacity 300ms ease-in-out;}
.stTabs [role="tablist"] {gap:.5rem;}
.stTabs [role="tab"] {padding:.6rem 1rem; border-radius:999px; border:1px solid #3a3a3a;}
.small {font-size:0.9rem; opacity:0.85;}
</style>
""", unsafe_allow_html=True)

st.title("KōreroNET • Daily Dashboard")

# ─────────────────────────────────────────────────────────────
# Paths & caches
# ─────────────────────────────────────────────────────────────
CACHE_ROOT   = Path("/tmp/koreronet_cache")
CSV_CACHE    = CACHE_ROOT / "csv"
CHUNK_CACHE  = CACHE_ROOT / "chunks"
for _p in (CSV_CACHE, CHUNK_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# Local fallback (Tab 1 only)
DEFAULT_ROOT = r"G:\My Drive\From the node"
ROOT_LOCAL   = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)

# ─────────────────────────────────────────────────────────────
# Drive client
# ─────────────────────────────────────────────────────────────
def _normalize_private_key(pk: str) -> str:
    if not isinstance(pk, str): return pk
    if "\\n" in pk: pk = pk.replace("\\n", "\n")
    if "-----BEGIN PRIVATE KEY-----" in pk and "-----END PRIVATE KEY-----" in pk:
        if "-----BEGIN PRIVATE KEY-----\n" not in pk:
            pk = pk.replace("-----BEGIN PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----\n", 1)
        if "\n-----END PRIVATE KEY-----" not in pk:
            pk = pk.replace("-----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----", 1)
    return pk

def build_drive():
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

GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID")
drive = build_drive() if GDRIVE_FOLDER_ID else None
DRIVE_ENABLED = bool(drive and GDRIVE_FOLDER_ID)

# ─────────────────────────────────────────────────────────────
# Drive helpers
# ─────────────────────────────────────────────────────────────
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
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
    from googleapiclient.http import MediaIoBaseDownload
    path.parent.mkdir(parents=True, exist_ok=True)
    req = drive.files().get_media(fileId=file_id)
    with open(path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return path

def ensure_csv_cached(meta: Dict[str, Any], subdir: str) -> Path:
    local_path = (CSV_CACHE / subdir / meta["name"])
    if not local_path.exists():
        download_to(local_path, meta["id"])
    return local_path

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
    local_path = CHUNK_CACHE / subdir / chunk_name
    if local_path.exists():
        return local_path
    # exact-name lookup within indicated folder
    for k in list_children(folder_id, max_items=2000):
        if k.get("name") == chunk_name:
            try:
                download_to(local_path, k["id"])
                return local_path
            except Exception:
                return None
    return None

# ─────────────────────────────────────────────────────────────
# Tab 1 helpers (root bn*/kn*)
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

# ─────────────────────────────────────────────────────────────
# Tab 2 helpers — MASTER CSV INDEX (FLEXIBLE)
# ─────────────────────────────────────────────────────────────
SNAP_RE = re.compile(r"^(\d{8})_(\d{6})$", re.IGNORECASE)

def _sanitize_label(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s

def _compose_chunk_name(kind: str, wav_base: str, start: float, end: float, label: str, conf: float) -> str:
    root = Path(wav_base).stem  # e.g., 19700102_011820
    start_s = f"{float(start):06.2f}"
    end_s   = f"{float(end):06.2f}"
    lab     = _sanitize_label(label)
    pc      = f"{float(conf):.2f}"
    tag     = "bn" if kind.lower()=="bn" else "kn"
    return f"{root}__{tag}_{start_s}_{end_s}__{lab}__p{pc}.wav"

def _parse_dt_from_wav(wav_base: str) -> Optional[datetime]:
    base = os.path.basename(wav_base)
    m = re.match(r"^(\d{8})_(\d{6})", base)
    if not m: return None
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")

def _find_chunk_folders(snapshot_id: str) -> Dict[str, str]:
    """Return {'KN': folder_id_or_snapshot, 'BN': folder_id_or_snapshot}."""
    kids = list_children(snapshot_id, max_items=2000)
    kn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "koreronet" in k.get("name","").lower()]
    bn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and "birdnet"   in k.get("name","").lower()]
    out = {}
    out["KN"] = (kn[0]["id"] if kn else snapshot_id)
    out["BN"] = (bn[0]["id"] if bn else snapshot_id)
    return out

def _match_master_name(name: str, kind: str) -> bool:
    n = name.lower()
    if kind == "KN":
        return (("koreronet" in n) and ("detect" in n) and n.endswith(".csv"))
    else:
        return (("birdnet" in n) and ("detect" in n) and n.endswith(".csv"))

def _find_master_csv_meta_anywhere(snapshot_id: str, kind: str) -> Optional[Dict[str, Any]]:
    # 1) root
    root_kids = list_children(snapshot_id, max_items=2000)
    root_files = [f for f in root_kids if f.get("mimeType") != "application/vnd.google-apps.folder"]
    candidates = [f for f in root_files if _match_master_name(f.get("name",""), kind)]
    if candidates:
        candidates.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
        return dict(candidates[0])
    # 2) one-level subfolders
    subfolders = [f for f in root_kids if f.get("mimeType") == "application/vnd.google-apps.folder"]
    for sf in subfolders:
        sub_files = list_children(sf["id"], max_items=2000)
        sub_files = [f for f in sub_files if f.get("mimeType") != "application/vnd.google-apps.folder"]
        cands = [f for f in sub_files if _match_master_name(f.get("name",""), kind)]
        if cands:
            cands.sort(key=lambda m: m.get("modifiedTime",""), reverse=True)
            return dict(cands[0])
    return None

@st.cache_data(show_spinner=True)
def list_snapshots_drive(root_folder_id: str) -> List[Dict[str, Any]]:
    # find Backup (case-insensitive)
    kids = list_children(root_folder_id, max_items=2000)
    backup = None
    for k in kids:
        if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name","").lower() == "backup":
            backup = k; break
    if not backup: return []
    snaps = [k for k in list_children(backup["id"], max_items=2000)
             if k.get("mimeType")=="application/vnd.google-apps.folder" and SNAP_RE.match(k.get("name",""))]
    snaps.sort(key=lambda m: m.get("name",""), reverse=True)
    return snaps

@st.cache_data(show_spinner=True)
def build_master_index(folder_id: str) -> pd.DataFrame:
    """
    Reads master CSVs (found flexibly) from each snapshot and returns:
    ['Date','Time','ActualTime','Kind','Label','Confidence','Start','End','WavBase',
     'ChunkName','ChunkDriveFolderId','SnapId']
    """
    rows: List[Dict[str,Any]] = []
    snaps = list_snapshots_drive(folder_id)
    for sn in snaps:
        snap_id = sn["id"]
        chunk_dirs = _find_chunk_folders(snap_id)

        # KN (koreronet) master
        kn_meta = _find_master_csv_meta_anywhere(snap_id, kind="KN")
        if kn_meta:
            kn_csv = ensure_csv_cached(kn_meta, subdir=f"snap_{snap_id}/koreronet")
            try:
                df = pd.read_csv(kn_csv)
                for _, r in df.iterrows():
                    wav = str(r.get("File",""))
                    t0 = _parse_dt_from_wav(wav)
                    if not t0: continue
                    start = float(r.get("Start", r.get("Start (s)", np.nan)))
                    end   = float(r.get("End", r.get("End (s)", np.nan)))
                    lab   = str(r.get("Label","Unknown"))
                    conf  = float(r.get("Confidence", np.nan))
                    at    = t0 + timedelta(seconds=float(start or 0.0))
                    rows.append({
                        "Date": at.date(),
                        "Time": at.time(),
                        "ActualTime": at,
                        "Kind": "KN",
                        "Label": lab,
                        "Confidence": conf,
                        "Start": start,
                        "End": end,
                        "WavBase": os.path.basename(wav),
                        "ChunkName": _compose_chunk_name("kn", wav, start, end, lab, conf),
                        "ChunkDriveFolderId": chunk_dirs["KN"],
                        "SnapId": snap_id,
                    })
            except Exception:
                pass

        # BN (birdnet) master
        bn_meta = _find_master_csv_meta_anywhere(snap_id, kind="BN")
        if bn_meta:
            bn_csv = ensure_csv_cached(bn_meta, subdir=f"snap_{snap_id}/birdnet")
            try:
                df = pd.read_csv(bn_csv)
                for _, r in df.iterrows():
                    wav = str(r.get("File",""))
                    t0 = _parse_dt_from_wav(os.path.basename(wav))
                    if not t0: continue
                    start = float(r.get("Start (s)", r.get("Start", np.nan)))
                    end   = float(r.get("End (s)",   r.get("End",   np.nan)))
                    lab   = str(r.get("Common name", r.get("Label","Unknown")))
                    conf  = float(r.get("Confidence", np.nan))
                    at    = t0 + timedelta(seconds=float(start or 0.0))
                    rows.append({
                        "Date": at.date(),
                        "Time": at.time(),
                        "ActualTime": at,
                        "Kind": "BN",
                        "Label": lab,
                        "Confidence": conf,
                        "Start": start,
                        "End": end,
                        "WavBase": os.path.basename(wav),
                        "ChunkName": _compose_chunk_name("bn", wav, start, end, lab, conf),
                        "ChunkDriveFolderId": chunk_dirs["BN"],
                        "SnapId": snap_id,
                    })
            except Exception:
                pass

    if not rows:
        return pd.DataFrame(columns=[
            "Date","Time","ActualTime","Kind","Label","Confidence","Start","End",
            "WavBase","ChunkName","ChunkDriveFolderId","SnapId"
        ])
    out = pd.DataFrame(rows)
    out.sort_values("ActualTime", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

# ─────────────────────────────────────────────────────────────
# Utility: calendar with safe snapping to available days
# ─────────────────────────────────────────────────────────────
def calendar_pick(available_days: List[date], label: str, help_txt: str = "") -> date:
    """
    Shows st.date_input with min/max bounds.
    If user selects a day with no data, snaps to nearest earlier day; if none earlier, to nearest later.
    """
    if not available_days:
        st.stop()
    available_days = sorted(available_days)
    d_min, d_max = available_days[0], available_days[-1]
    # default = latest with data
    d_val = st.date_input(label, value=d_max, min_value=d_min, max_value=d_max, help=help_txt)
    # Snap if necessary
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
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "📁 Drive"])

# =========================
# TAB 1 — Detections (root)
# =========================
with tab1:
    center = st.empty()
    with center.container():
        st.markdown('<div class="center-wrap fade"><div>🔎 Checking root CSVs…</div></div>', unsafe_allow_html=True)

    if DRIVE_ENABLED:
        bn_meta, kn_meta = list_csvs_drive_root(GDRIVE_FOLDER_ID)
    else:
        bn_meta, kn_meta = [], []

    if DRIVE_ENABLED and (bn_meta or kn_meta):
        center.empty(); center = st.empty()
        with center.container():
            st.markdown('<div class="center-wrap fade"><div>⬇️ Downloading root CSVs…</div></div>', unsafe_allow_html=True)
        bn_paths = [ensure_csv_cached(m, subdir="root/bn") for m in bn_meta]
        kn_paths = [ensure_csv_cached(m, subdir="root/kn") for m in kn_meta]
    else:
        bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
        bn_paths = [Path(p) for p in bn_local]
        kn_paths = [Path(p) for p in kn_local]

    center.empty(); center = st.empty()
    with center.container():
        st.markdown('<div class="center-wrap fade"><div>🧮 Building date index…</div></div>', unsafe_allow_html=True)

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

    # Calendar (with snapping)
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

    if st.button("Update", type="primary"):
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
# TAB 2 — Verify (MASTER CSV ONLY)
# ================================
with tab_verify:
    if not DRIVE_ENABLED:
        st.error("Google Drive is not configured in secrets."); st.stop()

    center2 = st.empty()
    with center2.container():
        st.markdown('<div class="center-wrap fade"><div>📚 Indexing master CSVs across snapshots…</div></div>', unsafe_allow_html=True)
    master = build_master_index(GDRIVE_FOLDER_ID)
    center2.empty()

    if master.empty:
        st.warning("No master CSVs found in any snapshot (searched root and one-level subfolders).")
        st.stop()

    colA, colB = st.columns([2,1])
    with colA:
        src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0)
    with colB:
        min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01)

    # Filter pool by source + confidence
    if src_mode_v == "KōreroNET (KN)":
        pool = master[master["Kind"]=="KN"]
    elif src_mode_v == "BirdNET (BN)":
        pool = master[master["Kind"]=="BN"]
    else:
        pool = master
    pool = pool[pd.to_numeric(pool["Confidence"], errors="coerce") >= float(min_conf_v)]
    if pool.empty:
        st.info("No rows above the selected confidence."); st.stop()

    # Calendar (with snapping) — built from available rows
    avail_days = sorted(pool["Date"].unique())
    day_pick = calendar_pick(avail_days, "Day", "Filter detections by calendar day.")

    day_df = pool[pool["Date"] == day_pick]
    if day_df.empty:
        st.warning("No detections for the chosen day."); st.stop()

    counts = day_df.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox(
        "Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_pick.isoformat()}::{src_mode_v}",
    )

    playlist = day_df[day_df["Label"] == species].sort_values("ActualTime").reset_index(drop=True)
    idx_key = f"v_idx::{day_pick.isoformat()}::{src_mode_v}::{species}"
    if idx_key not in st.session_state: st.session_state[idx_key] = 0
    idx = st.session_state[idx_key] % len(playlist)

    col1, col2, col3, col4 = st.columns([1,1,1,5])
    autoplay = False
    with col1:
        if st.button("⏮ Prev"): idx = (idx - 1) % len(playlist); autoplay = True
    with col2:
        if st.button("▶ Play"): autoplay = True
    with col3:
        if st.button("⏭ Next"): idx = (idx + 1) % len(playlist); autoplay = True
    st.session_state[idx_key] = idx
    with col4:
        st.markdown(f"**{species}** — {len(playlist)} detections on **{day_pick.isoformat()}** | index {idx+1}/{len(playlist)}")

    row = playlist.iloc[idx]
    st.markdown(f"**Confidence:** {float(row['Confidence']):.3f}  |  **Time:** {row['ActualTime']}")

    def _play_audio(row: pd.Series, auto: bool):
        chunk_name = str(row.get("ChunkName","") or "")
        folder_id  = str(row.get("ChunkDriveFolderId","") or "")
        kind       = str(row.get("Kind","UNK"))
        if not (chunk_name and folder_id):
            st.warning("No chunk mapping available."); return
        subdir = f"{kind}"
        with st.spinner("Fetching audio…"):
            cached = ensure_chunk_cached(chunk_name, folder_id, subdir=subdir)
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
# TAB 3 — Drive browser
# =====================
with tab3:
    if not DRIVE_ENABLED:
        st.info("Drive not configured. Set GDRIVE_FOLDER_ID and service_account in secrets.")
    else:
        st.caption("Top-level of Drive Folder ID")
        with st.spinner("Listing…"):
            kids = list_children(GDRIVE_FOLDER_ID, max_items=500)
        st.dataframe(
            [{"name": k.get("name"), "id": k.get("id"), "mimeType": k.get("mimeType"),
              "modifiedTime": k.get("modifiedTime"), "size": k.get("size")} for k in kids],
            use_container_width=True, hide_index=True
        )
