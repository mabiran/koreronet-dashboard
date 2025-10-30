#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KōreroNET Dashboard (with splash + Drive) — daily auto-refresh + change-aware indexing
# ------------------------------------------------------------
# - Daily cache key (Pacific/Auckland) forces a fresh index each day.
# - Drive-aware caching: re-downloads CSV if md5/modifiedTime changed.
# - Local-aware caching: re-indexes if file mtime/size changed.
# - No new buttons added; existing UI preserved.
# ------------------------------------------------------------

import os, io, re, glob, json, time, hashlib
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
# Daily cache key (Pacific/Auckland)
# ─────────────────────────────────────────────────────────────
def cache_day_key() -> str:
    return datetime.now(ZoneInfo("Pacific/Auckland")).strftime("%Y-%m-%d")

CACHE_DAY = cache_day_key()

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
# Secrets / Drive client
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
# Drive helpers (change-aware)
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

def _drive_list_signature(files: List[Dict[str, Any]]) -> str:
    """Stable signature over file set (id+md5+modifiedTime)."""
    sigsrc = "\n".join(f"{f.get('id','')}|{f.get('md5Checksum','')}|{f.get('modifiedTime','')}" for f in sorted(files, key=lambda x: x.get("name","")))
    return hashlib.sha1(sigsrc.encode("utf-8", errors="ignore")).hexdigest()

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

def ensure_csv_cached(meta: Dict[str, Any], subdir: str) -> Path:
    """
    Content-aware: re-download if md5 changed (or no md5 but modifiedTime changed).
    """
    local_dir = CSV_CACHE / subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / meta["name"]
    md5 = meta.get("md5Checksum") or ""
    mtime = meta.get("modifiedTime") or ""
    sig_path = local_path.with_suffix(local_path.suffix + ".sig")

    need = not local_path.exists()
    old_sig = sig_path.read_text(errors="ignore").strip() if sig_path.exists() else ""
    new_sig = f"{md5}|{mtime}"

    if old_sig != new_sig:
        need = True

    if need:
        download_to(local_path, meta["id"])
        sig_path.write_text(new_sig)

    return local_path

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
    local_path = CHUNK_CACHE / subdir / chunk_name
    if local_path.exists():
        return local_path
    for k in list_children(folder_id, max_items=2000):
        if k.get("name") == chunk_name:
            try:
                download_to(local_path, k["id"])
                return local_path
            except Exception:
                return None
    return None

def find_subfolder_by_name(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
    kids = list_children(root_id, max_items=2000)
    for k in kids:
        if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name","").lower()==name_ci.lower():
            return k
    return None

# ─────────────────────────────────────────────────────────────
# Tab 1 (root CSV heatmap) — change-aware indexing
# ─────────────────────────────────────────────────────────────
def _file_sig_local(p: Path) -> str:
    try:
        stt = p.stat()
        return f"{p.name}|{int(stt.st_mtime)}|{stt.st_size}"
    except Exception:
        return f"{p.name}|missing|0"

@st.cache_data(show_spinner=False)
def list_csvs_local(root: str, cache_day: str) -> Tuple[List[str], List[str], str, str]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    # Build signatures so downstream caches rerun when files change
    bn_sig = hashlib.sha1("\n".join(_file_sig_local(Path(p)) for p in bn_paths).encode()).hexdigest()
    kn_sig = hashlib.sha1("\n".join(_file_sig_local(Path(p)) for p in kn_paths).encode()).hexdigest()
    return bn_paths, kn_paths, bn_sig, kn_sig

@st.cache_data(show_spinner=False)
def list_csvs_drive_root(folder_id: str, cache_day: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    kids = list_children(folder_id, max_items=2000)
    bn = [k for k in kids if k.get("name","").lower().startswith("bn") and k.get("name","").lower().endswith(".csv")]
    kn = [k for k in kids if k.get("name","").lower().startswith("kn") and k.get("name","").lower().endswith(".csv")]
    bn.sort(key=lambda m: m.get("name","")); kn.sort(key=lambda m: m.get("name",""))
    # Signatures to trigger re-index when Drive content changes
    bn_sig = _drive_list_signature(bn)
    kn_sig = _drive_list_signature(kn)
    return bn, kn, bn_sig, kn_sig

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
def build_date_index(paths: List[str], path_sigs: List[str], cache_day: str) -> Dict[date, List[str]]:
    """
    Rebuilds if any path signature changes OR a new day starts.
    """
    idx: Dict[date, List[str]] = {}
    for p in paths:
        pth = str(p)
        try:
            for chunk in pd.read_csv(pth, usecols=["ActualTime"], chunksize=5000):
                s = pd.to_datetime(chunk["ActualTime"], errors="coerce", dayfirst=True)
                for ts in s.dropna():
                    idx.setdefault(ts.date(), []).append(pth)
        except Exception:
            try:
                for chunk in pd.read_csv(pth, chunksize=5000):
                    if "ActualTime" not in chunk.columns: continue
                    s = pd.to_datetime(chunk["ActualTime"], errors="coerce", dayfirst=True)
                    for ts in s.dropna():
                        idx.setdefault(ts.date(), []).append(pth)
            except Exception:
                continue
    return idx

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

@st.cache_data(show_spinner=False)
def load_csv(path: str | Path, cache_day: str) -> pd.DataFrame:
    return pd.read_csv(str(path))

# ─────────────────────────────────────────────────────────────
# Snapshot/master logic for Tab 2 (daily refresh baked in)
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
def build_master_index_by_snapshot_date(root_folder_id: str, cache_day: str) -> pd.DataFrame:
    backup = _find_backup_folder(root_folder_id)
    if not backup:
        return pd.DataFrame(columns=[
            "Date","Kind","Label","Confidence","Start","End","WavBase","ChunkName",
            "ChunkDriveFolderId","SnapId","SnapName"
        ])
    snaps = [k for k in list_children(backup["id"], max_items=2000)
             if k.get("mimeType")=="application/vnd.google-apps.folder" and SNAP_RE.match(k.get("name",""))]
    snaps.sort(key=lambda m: m.get("name",""), reverse=True)

    # Signature over snapshot contents to re-run if masters change
    master_sig_src = []
    for sn in snaps:
        snap_id = sn["id"]
        kids = list_children(snap_id, max_items=2000)
        for f in kids:
            master_sig_src.append(f"{f.get('id','')}|{f.get('md5Checksum','')}|{f.get('modifiedTime','')}")
    _ = hashlib.sha1("\n".join(sorted(master_sig_src)).encode()).hexdigest()  # unused var, but affects cache via arg binding

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
    center = st.empty()
    with center.container():
        st.markdown('<div class="center-wrap fade-enter"><div>🔎 Checking root CSVs…</div></div>', unsafe_allow_html=True)

    if drive_enabled():
        bn_meta, kn_meta, bn_list_sig, kn_list_sig = list_csvs_drive_root(GDRIVE_FOLDER_ID, CACHE_DAY)
    else:
        bn_meta, kn_meta, bn_list_sig, kn_list_sig = [], [], "", ""

    if drive_enabled() and (bn_meta or kn_meta):
        center.empty(); center = st.empty()
        with center.container():
            st.markdown('<div class="center-wrap"><div>⬇️ Syncing root CSVs…</div></div>', unsafe_allow_html=True)
        bn_paths = [ensure_csv_cached(m, subdir="root/bn") for m in bn_meta]
        kn_paths = [ensure_csv_cached(m, subdir="root/kn") for m in kn_meta]
        # After download, compute local sigs (mtime/size) to drive re-indexing
        bn_path_sigs = [_file_sig_local(Path(p)) for p in bn_paths]
        kn_path_sigs = [_file_sig_local(Path(p)) for p in kn_paths]
    else:
        bn_local, kn_local, _bn_sig_local, _kn_sig_local = list_csvs_local(ROOT_LOCAL, CACHE_DAY)
        bn_paths = [Path(p) for p in bn_local]
        kn_paths = [Path(p) for p in kn_local]
        bn_path_sigs = [_file_sig_local(p) for p in bn_paths]
        kn_path_sigs = [_file_sig_local(p) for p in kn_paths]

    center.empty(); center = st.empty()
    with center.container():
        st.markdown('<div class="center-wrap"><div>🧮 Building date index…</div></div>', unsafe_allow_html=True)

    bn_by_date = build_date_index([str(p) for p in bn_paths], bn_path_sigs, CACHE_DAY) if bn_paths else {}
    kn_by_date = build_date_index([str(p) for p in kn_paths], kn_path_sigs, CACHE_DAY) if kn_paths else {}

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
                df = load_csv(p, CACHE_DAY)
                std = standardize_df(df, kind)
                std = std[std["ActualTime"].dt.date == day_selected]
                if not std.empty: frames.append(std)
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Auto-render (no extra button)
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

    center2 = st.empty()
    with center2.container():
        st.markdown('<div class="center-wrap fade-enter"><div>📚 Indexing master CSVs by snapshot date…</div></div>', unsafe_allow_html=True)
    master = build_master_index_by_snapshot_date(GDRIVE_FOLDER_ID, CACHE_DAY)
    center2.empty()

    if master.empty:
        st.warning("No master CSVs found in any snapshot (looked in snapshot root and one-level subfolders).")
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
    day_pick = calendar_pick(list(avail_days), "Day", "Dates come from snapshot folder names under Backup/.")

    day_df = pool[pool["Date"] == day_pick]
    if day_df.empty:
        st.warning("No detections for the chosen date."); st.stop()

    counts = day_df.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox(
        "Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_pick.isoformat()}::{src_mode_v}",
    )

    playlist = day_df[day_df["Label"] == species].sort_values(["WavBase","Kind"]).reset_index(drop=True)

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
    st.markdown(f"**Date:** {row['Date']}  |  **File:** `{row['ChunkName']}`  |  **Kind:** {row['Kind']}  |  **Confidence:** {float(row['Confidence']):.3f}")

    def _play_audio(row_: pd.Series, auto: bool):
        chunk_name = str(row_.get("ChunkName","") or "")
        folder_id  = str(row_.get("ChunkDriveFolderId","") or "")
        kind       = str(row_.get("Kind","UNK"))
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
# TAB 3 — Power graph
# =====================
with tab3:
    st.subheader("Node Power History")
    if not drive_enabled():
        st.error("Google Drive is not configured in secrets."); st.stop()

    logs_folder = find_subfolder_by_name(GDRIVE_FOLDER_ID, "Power logs")
    if not logs_folder:
        st.warning("Could not find 'Power logs' folder under the Drive root.")
        st.stop()

    LOG_RE = re.compile(r"^power_history_(\d{8})_(\d{6})\.log$", re.IGNORECASE)

    @st.cache_data(show_spinner=True)
    def list_power_logs(folder_id: str, cache_day: str) -> List[Dict[str, Any]]:
        kids = list_children(folder_id, max_items=2000)
        files = [k for k in kids if k.get("mimeType") != "application/vnd.google-apps.folder" and LOG_RE.match(k.get("name",""))]
        files.sort(key=lambda m: m.get("name",""), reverse=True)
        return files

    @st.cache_data(show_spinner=False)
    def ensure_log_cached(meta: Dict[str, Any], cache_day: str) -> Path:
        local_path = POWER_CACHE / meta["name"]
        sig_path = local_path.with_suffix(".sig")
        new_sig = f"{meta.get('md5Checksum','')}|{meta.get('modifiedTime','')}"
        old_sig = sig_path.read_text(errors="ignore").strip() if sig_path.exists() else ""
        if (not local_path.exists()) or (new_sig != old_sig):
            download_to(local_path, meta["id"])
            sig_path.write_text(new_sig)
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
    def build_power_timeseries(folder_id: str, latest_n: int, cache_day: str) -> pd.DataFrame:
        files = list_power_logs(folder_id, cache_day)
        if not files:
            return pd.DataFrame(columns=["t","PH_WH","PH_mAh","PH_SoCi","PH_SoCv"])
        frames: List[pd.DataFrame] = []
        for meta in files[:max(1, latest_n)]:
            local = ensure_log_cached(meta, cache_day)
            df = parse_power_log(local)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["t","PH_WH","PH_mAh","PH_SoCi","PH_SoCv"])
        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values("t", inplace=True)
        merged = merged.drop_duplicates(subset=["t"], keep="last")
        merged.reset_index(drop=True, inplace=True)
        return merged

    cols = st.columns([2,1])
    with cols[0]:
        lookback = st.number_input("How many latest logs to stitch", min_value=1, max_value=50, value=2, step=1, key="tab3_lookback")
    with cols[1]:
        pass

    # Auto-build (no extra button)
    with st.spinner("Building power time-series…"):
        ts = build_power_timeseries(logs_folder["id"], latest_n=int(lookback), cache_day=CACHE_DAY)

    if ts.empty:
        st.warning("No parsable power logs found.")
    else:
        fig = go.Figure()
        # SoC_i (%) on y1
        fig.add_trace(go.Scatter(
            x=ts["t"], y=ts["PH_SoCi"], mode="lines", name="SoC_i (%)", yaxis="y1"
        ))
        # Wh on y2
        fig.add_trace(go.Scatter(
            x=ts["t"], y=ts["PH_WH"], mode="lines", name="Energy (Wh)", yaxis="y2"
        ))

        fig.update_layout(
            title="Power / State of Charge over time",
            xaxis=dict(title="Time"),
            yaxis=dict(title="SoC (%)", range=[0,100]),
            yaxis2=dict(title="Wh", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quick last-point indicators
        last = ts.iloc[-1]
        c1, c2 = st.columns(2)
        with c1: st.metric("Last SoC_i (%)", f"{last['PH_SoCi']:.1f}")
        with c2: st.metric("Last Energy (Wh)", f"{last['PH_WH']:.2f}")
