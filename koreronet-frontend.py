#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KōreroNET Dashboard (with splash + Drive)
# ------------------------------------------------------------
# - Splash screen (“KōreroNET” + “AUT”) for ~2s, then main UI
# - Tab 1: Detections from timestamped master CSVs in “From the node”
# - Tab 2: Verify/play chunks using master CSV rows (Clip, Label, Probability, ActualStartTime)
# - Tab 3: Power graph; zeros treated as restarts -> removed & interpolated hourly
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

NEW_PIPELINE_START = date(2025, 10, 31)

# ---- Chunk filename parsing (for fuzzy-match fallback) ----
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

DEFAULT_ROOT = r"G:\My Drive\From the node"  # local mirror; optional
ROOT_LOCAL   = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)

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

def ensure_csv_cached(meta: Dict[str, Any], subdir: str) -> Path:
    local_path = (CSV_CACHE / subdir / meta["name"])
    if not local_path.exists():
        download_to(local_path, meta["id"])
    return local_path

# ========== New: Master CSV discovery ==========
MASTER_BIRDNET_SUFFIX   = "_birdnet_master.csv"
MASTER_KORERONET_SUFFIX = "_koreronet_master.csv"

@st.cache_data(show_spinner=False)
def list_master_csvs_drive_root(folder_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kids = list_children(folder_id, max_items=2000)
    bn = [k for k in kids if k.get("name","").lower().endswith(MASTER_BIRDNET_SUFFIX)]
    kn = [k for k in kids if k.get("name","").lower().endswith(MASTER_KORERONET_SUFFIX)]
    bn.sort(key=lambda m: m.get("name",""))
    kn.sort(key=lambda m: m.get("name",""))
    return bn, kn

@st.cache_data(show_spinner=False)
def list_master_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, f"*{MASTER_BIRDNET_SUFFIX}")))
    kn_paths = sorted(glob.glob(os.path.join(root, f"*{MASTER_KORERONET_SUFFIX}")))
    return bn_paths, kn_paths

# ========== Backup folders for audio chunks ==========
def find_subfolder_by_name(root_id: str, name_ci: str) -> Optional[Dict[str, Any]]:
    kids = list_children(root_id, max_items=2000)
    for k in kids:
        if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name","").lower()==name_ci.lower():
            return k
    return None

def find_backup_audio_folders(root_id: str) -> Dict[str,str]:
    """
    Returns drive IDs for backup audio subfolders:
      {"BN": <id of Backup/birdnet>, "KN": <id of Backup/koreronet>}
    Falls back to Backup/ root if a subfolder is missing.
    """
    backup = find_subfolder_by_name(root_id, "Backup")
    if not backup:
        return {"BN": root_id, "KN": root_id}
    kids = list_children(backup["id"], max_items=2000)
    id_bn = next((k["id"] for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name","").lower()=="birdnet"), backup["id"])
    id_kn = next((k["id"] for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name","").lower()=="koreronet"), backup["id"])
    return {"BN": id_bn, "KN": id_kn}

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
    """
    Try exact name first, then fuzzy-match by parsed parts if necessary.
    """
    local_path = CHUNK_CACHE / subdir / chunk_name
    if local_path.exists():
        return local_path

    kids = list_children(folder_id)
    name_to_id = {k.get("name"): k.get("id") for k in kids}
    if chunk_name in name_to_id:
        try:
            download_to(local_path, name_to_id[chunk_name])
            return local_path
        except Exception:
            return None

    # Fuzzy fallback
    target = _parse_chunk_filename(chunk_name)
    if not target:
        return None
    root_t, tag_t, s_t, e_t, label_t = target["root"], target["tag"], target["s"], target["e"], target["label"].lower()

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
        download_to(local_path, best_id)
        st.caption(f"⚠️ Used fuzzy match: requested `{chunk_name}` → found `{best_name}`")
        return local_path
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# Master CSV utilities (new format)
# ─────────────────────────────────────────────────────────────
def _read_master_standard(path: Path) -> pd.DataFrame:
    """
    Expected columns in master CSVs:
      Clip, ActualStartTime, Label, Probability
    """
    df = pd.read_csv(str(path))
    # tolerate small header variations
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        return None
    c_clip = pick("Clip","chunk","file","wav","chunkname")
    c_time = pick("ActualStartTime","ActualTime","Time","Timestamp")
    c_label= pick("Label","Common name","Common_name","Species","Class")
    c_prob = pick("Probability","Confidence","Prob","P")
    if not c_clip or not c_time or not c_label:
        return pd.DataFrame(columns=["Clip","ActualStartTime","Label","Probability"])
    out = pd.DataFrame({
        "Clip": df[c_clip].astype(str),
        "ActualStartTime": pd.to_datetime(df[c_time], errors="coerce", dayfirst=True),
        "Label": df[c_label].astype(str).str.strip(),
        "Probability": pd.to_numeric(df.get(c_prob, np.nan), errors="coerce")
    })
    # Drop Noise if present (defensive, process.py already ignores)
    out = out[out["Label"].str.lower() != "noise"]
    out = out.dropna(subset=["ActualStartTime","Label"])
    return out

@st.cache_data(show_spinner=False)
def build_date_index_from_masters(paths: List[Path]) -> Dict[date, List[Path]]:
    idx: Dict[date, List[Path]] = {}
    for p in paths:
        try:
            for chunk in pd.read_csv(p, usecols=["ActualStartTime"], chunksize=5000):
                s = pd.to_datetime(chunk["ActualStartTime"], errors="coerce", dayfirst=True)
                for d in s.dropna().dt.date.unique():
                    idx.setdefault(d, []).append(p)
        except Exception:
            try:
                df = _read_master_standard(p)
                for d in df["ActualStartTime"].dt.date.unique():
                    idx.setdefault(d, []).append(p)
            except Exception:
                pass
    # dedup lists
    for k in list(idx.keys()):
        idx[k] = sorted(set(idx[k]), key=lambda p: p.name)
    return idx

def load_master_for_day(paths: List[Path], day: date) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = _read_master_standard(p)
            df = df[df["ActualStartTime"].dt.date == day]
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Clip","ActualStartTime","Label","Probability"])

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "⚡ Power"])

# =========================
# TAB 1 — Detections (masters)
# =========================
with tab1:
    center = st.empty()
    with center.container():
        st.markdown('<div class="center-wrap fade-enter"><div>🔎 Scanning for master CSVs…</div></div>', unsafe_allow_html=True)

    # Discover master CSVs in root
    if drive_enabled():
        bn_meta, kn_meta = list_master_csvs_drive_root(GDRIVE_FOLDER_ID)
        bn_paths = [ensure_csv_cached(m, subdir="root_master/bn") for m in bn_meta]
        kn_paths = [ensure_csv_cached(m, subdir="root_master/kn") for m in kn_meta]
    else:
        bn_local, kn_local = list_master_csvs_local(ROOT_LOCAL)
        bn_paths = [Path(p) for p in bn_local]
        kn_paths = [Path(p) for p in kn_local]

    center.empty()
    if not (bn_paths or kn_paths):
        st.warning("No master CSVs found in the root.")
        st.stop()

    bn_index = build_date_index_from_masters(bn_paths) if bn_paths else {}
    kn_index = build_date_index_from_masters(kn_paths) if kn_paths else {}

    src = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01)

    if src == "Combined":
        days = sorted(set(bn_index.keys()).union(set(kn_index.keys())))
        help_txt = "Dates appearing in either BN or KN masters."
    elif src == "BirdNET (BN)":
        days = sorted(bn_index.keys()); help_txt = "Dates present in any BN master."
    else:
        days = sorted(kn_index.keys()); help_txt = "Dates present in any KN master."

    if not days:
        st.warning(f"No available dates for {src}."); st.stop()

    d = st.date_input("Day", value=days[-1], min_value=days[0], max_value=days[-1], help=help_txt)

    def make_heatmap(df: pd.DataFrame, title: str):
        if df.empty:
            st.warning("No detections for this day."); return
        # Apply confidence
        df = df[pd.to_numeric(df["Probability"], errors="coerce") >= float(min_conf)]
        if df.empty:
            st.info("No detections after applying the confidence filter."); return
        df["Hour"] = df["ActualStartTime"].dt.hour
        hour_labels = {h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)}
        order = [hour_labels[h] for h in range(24)]
        df["HourLabel"] = df["Hour"].map(hour_labels)
        pivot = df.groupby(["Label","HourLabel"]).size().unstack(fill_value=0).astype(int)
        for lbl in order:
            if lbl not in pivot.columns: pivot[lbl] = 0
        pivot = pivot[order]
        totals = pivot.sum(axis=1)
        pivot = pivot.loc[totals.sort_values(ascending=False).index]
        fig = px.imshow(
            pivot.values, x=pivot.columns, y=pivot.index,
            color_continuous_scale="RdYlBu_r",
            labels=dict(x="Hour (AM/PM)", y="Species / label", color="Detections"),
            text_auto=True, aspect="auto", title=title,
        )
        fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Show results", type="primary", key="tab1_show_btn"):
        with st.spinner("Rendering heatmap…"):
            if src == "BirdNET (BN)":
                df = load_master_for_day(bn_index.get(d, []), d)
                make_heatmap(df, f"BirdNET • {d.isoformat()}")
            elif src == "KōreroNET (KN)":
                df = load_master_for_day(kn_index.get(d, []), d)
                make_heatmap(df, f"KōreroNET • {d.isoformat()}")
            else:
                df_bn = load_master_for_day(bn_index.get(d, []), d)
                df_kn = load_master_for_day(kn_index.get(d, []), d)
                make_heatmap(pd.concat([df_bn, df_kn], ignore_index=True), f"Combined (BN+KN) • {d.isoformat()}")

# ================================
# TAB 2 — Verify (masters → audio)
# ================================
with tab_verify:
    if not drive_enabled():
        st.error("Google Drive is not configured in secrets."); st.stop()

    center2 = st.empty()
    with center2.container():
        st.markdown('<div class="center-wrap fade-enter"><div>📚 Loading master CSVs…</div></div>', unsafe_allow_html=True)

    # Load all masters (root)
    bn_meta, kn_meta = list_master_csvs_drive_root(GDRIVE_FOLDER_ID)
    bn_paths = [ensure_csv_cached(m, subdir="root_master/bn") for m in bn_meta]
    kn_paths = [ensure_csv_cached(m, subdir="root_master/kn") for m in kn_meta]
    df_bn = pd.concat([_read_master_standard(p) for p in bn_paths], ignore_index=True) if bn_paths else pd.DataFrame(columns=["Clip","ActualStartTime","Label","Probability"])
    df_kn = pd.concat([_read_master_standard(p) for p in kn_paths], ignore_index=True) if kn_paths else pd.DataFrame(columns=["Clip","ActualStartTime","Label","Probability"])
    center2.empty()

    if df_bn.empty and df_kn.empty:
        st.warning("No master CSVs found in the root.")
        st.stop()

    # Choose source & date
    colA, colB = st.columns([2,1])
    with colA:
        src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0)
    with colB:
        min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01)

    if src_mode_v == "KōreroNET (KN)":
        pool = df_kn
    elif src_mode_v == "BirdNET (BN)":
        pool = df_bn
    else:
        pool = pd.concat([df_kn.assign(Kind="KN"), df_bn.assign(Kind="BN")], ignore_index=True)
        if "Kind" not in pool.columns:
            pool["Kind"] = np.where(pool.index < len(df_kn), "KN", "BN")

    pool = pool[pd.to_numeric(pool["Probability"], errors="coerce") >= float(min_conf_v)]
    if pool.empty:
        st.info("No rows above the selected confidence."); st.stop()

    avail_days = sorted(pool["ActualStartTime"].dt.date.unique())
    day_pick = st.date_input("Day", value=avail_days[-1], min_value=avail_days[0], max_value=avail_days[-1])

    day_df = pool[pool["ActualStartTime"].dt.date == day_pick]
    if day_df.empty:
        st.warning("No detections for the chosen date."); st.stop()

    # Species list with counts
    counts = day_df.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox(
        "Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_pick.isoformat()}::{src_mode_v}",
    )

    playlist = day_df[day_df["Label"] == species].copy()
    # Attach Kind if missing (when user chose single source)
    if "Kind" not in playlist.columns:
        playlist["Kind"] = "KN" if src_mode_v.startswith("K") else "BN"
    playlist.sort_values(["ActualStartTime","Clip","Kind"], inplace=True, ignore_index=True)

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
    st.markdown(
        f"**Date:** {row['ActualStartTime'].date()}  |  "
        f"**File:** `{row['Clip']}`  |  "
        f"**Kind:** {row['Kind']}  |  "
        f"**Confidence:** {float(row['Probability']):.3f}"
    )

    # Resolve backup folders for audio (BN/KN)
    bk = find_backup_audio_folders(GDRIVE_FOLDER_ID)

    def _play_audio_from_backup(row_: pd.Series, auto: bool):
        chunk_name = str(row_.get("Clip","") or "")
        kind       = str(row_.get("Kind","UNK")).upper()
        folder_id  = bk.get(kind, None)
        if not (chunk_name and folder_id):
            st.warning("No backup folder mapping available."); return
        with st.spinner("Fetching audio…"):
            cached = ensure_chunk_cached(chunk_name, folder_id, subdir=kind)
        if not cached or not cached.exists():
            st.warning("Audio chunk not found in Drive backup."); return
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

    _play_audio_from_backup(row, autoplay)

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

        # Header timestamp (first line or first parseable datetime)
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

        # Build time axis: last element corresponds to head_dt; hourly cadence
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

        # New rule: zeros mean "node restarted" → treat as NaN, then hourly interpolate.
        for c in ["PH_WH", "PH_mAh", "PH_SoCi", "PH_SoCv"]:
            merged.loc[merged[c] == 0.0, c] = np.nan

        merged.set_index("t", inplace=True)
        # Force hourly frequency across any gaps (including day boundaries)
        merged = merged.asfreq("H")
        # Interpolate linearly across NaNs introduced by zeros or missing hours
        merged = merged.interpolate(method="time", limit_direction="both")
        merged.reset_index(inplace=True)

        return merged

    cols = st.columns([2,1])
    with cols[0]:
        lookback = st.number_input("How many latest logs to stitch", min_value=1, max_value=50, value=2, step=1, key="tab3_lookback")
    with cols[1]:
        pass

    if st.button("Show results", type="primary", key="tab3_show_btn"):
        with st.spinner("Building power time-series…"):
            ts = build_power_timeseries(logs_folder["id"], latest_n=int(lookback))

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
