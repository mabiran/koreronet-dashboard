#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KōreroNET Dashboard — Google Drive backed, with visible progress + throttling

- Uses st.secrets: GDRIVE_FOLDER_ID and [service_account] (or SERVICE_ACCOUNT_JSON).
- Tab 1: heatmaps from bn*/kn* CSVs at the root of Folder ID (downloads on demand).
- Tab 2: verify snapshots under Backup/YYYYMMDD_HHMMSS/{koreronet,birdnet} (progress shown).
- Tab 3: Drive browser and cache controls.
"""

import os, io, glob, re, json, shutil
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- UI ----------
st.set_page_config(page_title="KōreroNET Dashboard", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:1rem;}
.stTabs [role="tablist"] {gap: .5rem;}
.stTabs [role="tab"] {padding: .6rem 1rem; border-radius: 999px; border: 1px solid #3a3a3a;}
</style>
""", unsafe_allow_html=True)
st.title("KōreroNET • Daily Dashboard")

# ---------- Local caches ----------
CACHE_ROOT = Path("/tmp/koreronet_cache")
CSV_CACHE = CACHE_ROOT / "csv"
CHUNK_CACHE = CACHE_ROOT / "chunks"
for p in (CSV_CACHE, CHUNK_CACHE):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Drive client ----------
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
        sa_tbl = st.secrets.get("service_account")
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
    except Exception as e:
        st.error(f"Drive client error: {e}")
        return None

GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID")
drive = build_drive() if GDRIVE_FOLDER_ID else None
DRIVE_ENABLED = bool(drive and GDRIVE_FOLDER_ID)

# ---------- Drive helpers ----------
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
    items, token = [], None
    while True:
        page_size = min(100, max_items - len(items))
        if page_size <= 0: break
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,md5Checksum)",
            pageSize=page_size,
            pageToken=token,
            orderBy="folder,name_natural",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token: break
    return items

def find_child_folder_by_name(parent_id: str, name: str) -> Optional[Dict[str, Any]]:
    for k in list_children(parent_id, max_items=2000):
        if k.get("mimeType") == "application/vnd.google-apps.folder" and k.get("name") == name:
            return k
    return None

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

def ensure_csv_cached_by_meta(meta: Dict[str, Any], subdir: str = "") -> Path:
    name = meta["name"]
    target_dir = CSV_CACHE / subdir if subdir else CSV_CACHE
    local_path = target_dir / name
    if not local_path.exists():
        download_to(local_path, meta["id"])
    return local_path

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
    local_path = (CHUNK_CACHE / subdir / chunk_name)
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

# ---------- Local fallback roots ----------
DEFAULT_ROOT = r"G:\My Drive\From the node"
BACKUP_ROOT_DEFAULT = r"G:\My Drive\From the node\Backup"
ROOT_LOCAL = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)
BACKUP_ROOT_LOCAL = os.getenv("KORERONET_BACKUP_ROOT", BACKUP_ROOT_DEFAULT)

# ---------- CSV discovery ----------
@st.cache_data(show_spinner=False)
def list_csvs_drive(folder_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kids = list_children(folder_id, max_items=2000)
    bn = [k for k in kids if k.get("name","").lower().startswith("bn") and k.get("name","").lower().endswith(".csv")]
    kn = [k for k in kids if k.get("name","").lower().startswith("kn") and k.get("name","").lower().endswith(".csv")]
    bn.sort(key=lambda m: m.get("name",""))
    kn.sort(key=lambda m: m.get("name",""))
    return bn, kn

@st.cache_data(show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    return bn_paths, kn_paths

def _download_and_return_paths(metas: List[Dict[str, Any]], label: str, subdir: str = "") -> List[Path]:
    out = []
    prog = st.progress(0.0)
    stat = st.empty()
    n = max(1, len(metas))
    for i, m in enumerate(metas, 1):
        stat.text(f"Downloading {label} {i}/{n}: {m.get('name')}")
        out.append(ensure_csv_cached_by_meta(m, subdir=subdir))
        prog.progress(i/n)
    stat.text("Done.")
    prog.progress(1.0)
    return out

# ---------- Date indexing ----------
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

# ---------- Normalize + heatmap ----------
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
        if lbl not in pivot.columns: pivot[lbl] = 0
    pivot = pivot[hour_order]
    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values(ascending=False).index]
    if pivot.empty:
        st.warning("No data to plot."); return
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

# ---------- Snapshots ----------
SNAP_RE = re.compile(r"^(\d{8})_(\d{6})$")
def _parse_snapshot(name: str) -> Optional[datetime]:
    m = SNAP_RE.match(name)
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S") if m else None

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
    backup = find_child_folder_by_name(root_folder_id, "Backup")
    if not backup: return []
    kids = [k for k in list_children(backup["id"], max_items=2000)
            if k.get("mimeType") == "application/vnd.google-apps.folder" and SNAP_RE.match(k.get("name",""))]
    out = []
    for k in kids:
        dt = _parse_snapshot(k["name"])
        if dt: out.append((dt, k["name"], k["id"]))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def list_snapshots_any() -> Tuple[List[Tuple[datetime, str, str]], str]:
    if DRIVE_ENABLED:
        return list_snapshots_drive(GDRIVE_FOLDER_ID), "drive"
    else:
        return list_snapshots_local(BACKUP_ROOT_LOCAL), "local"

# ---------- App Tabs ----------
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "📁 Drive"])

# ===== TAB 1 =====
with tab1:
    st.caption("Uses CSVs at the **root** of your Google Drive Folder ID (bn*.csv / kn*.csv).")
    max_root_csv = st.slider("Max CSVs per source (root)", 10, 1000, 200, 10)
    if DRIVE_ENABLED:
        with st.status("Listing root CSVs…", expanded=False) as s:
            bn_meta, kn_meta = list_csvs_drive(GDRIVE_FOLDER_ID)
            s.update(label=f"Found BN={len(bn_meta)}, KN={len(kn_meta)}")
        if not bn_meta and not kn_meta:
            st.error("No bn_*.csv / kn_*.csv at root of the Drive folder."); st.stop()
        bn_meta = bn_meta[:max_root_csv]
        kn_meta = kn_meta[:max_root_csv]
        st.write(f"Indexing up to BN={len(bn_meta)}, KN={len(kn_meta)}")
        st.subheader("Indexing dates")
        with st.expander("Download progress (root CSVs)", expanded=True):
            bn_paths = _download_and_return_paths(bn_meta, "BN", subdir="root/bn") if bn_meta else []
            kn_paths = _download_and_return_paths(kn_meta, "KN", subdir="root/kn") if kn_meta else []
        with st.status("Building date index…", expanded=False):
            bn_by_date = build_date_index(bn_paths)
            kn_by_date = build_date_index(kn_paths)
    else:
        bn_paths_local, kn_paths_local = list_csvs_local(ROOT_LOCAL)
        if not bn_paths_local and not kn_paths_local:
            st.error("No bn_*.csv / kn_*.csv found on local fallback."); st.stop()
        bn_paths = [Path(p) for p in bn_paths_local][:max_root_csv]
        kn_paths = [Path(p) for p in kn_paths_local][:max_root_csv]
        bn_by_date = build_date_index(bn_paths)
        kn_by_date = build_date_index(kn_paths)

    bn_dates = sorted(bn_by_date.keys())
    kn_dates = sorted(kn_by_date.keys())
    paired_dates = sorted(set(bn_dates).intersection(set(kn_dates)))

    src = st.selectbox("Source", ["KōreroNET (kn)", "BirdNET (bn)", "Combined"], index=0)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01)

    if src == "Combined":
        options, help_txt = paired_dates, "Only dates that have BOTH BN & KN detections."
    elif src == "BirdNET (bn)":
        options, help_txt = bn_dates, "Dates that appear inside any BN file."
    else:
        options, help_txt = kn_dates, "Dates that appear inside any KN file."

    if not options:
        st.warning(f"No available dates for {src}."); st.stop()

    d_default = options[-1]
    d = st.date_input("Day", value=d_default, min_value=options[0], max_value=options[-1], help=help_txt)
    if d not in options:
        earlier = [x for x in options if x <= d]
        d = (earlier[-1] if earlier else options[0])
        st.info(f"No data for the chosen date; showing {d.isoformat()}.")

    def load_and_filter(paths: List[Path], kind: str, day_selected: date):
        frames = []
        with st.status("Loading CSVs for selected day…", expanded=False):
            for p in paths:
                try:
                    df = load_csv(p)
                    std = standardize_df(df, kind)
                    std = std[std["ActualTime"].dt.date == day_selected]
                    if not std.empty: frames.append(std)
                except Exception:
                    pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    with st.form(key="det_form"):
        if st.form_submit_button("Update"):
            if src == "BirdNET (bn)":
                df_bn = load_and_filter([Path(p) for p in bn_by_date.get(d, [])], "bn", d)
                make_heatmap(df_bn, min_conf, f"BirdNET • {d.isoformat()}")
            elif src == "KōreroNET (kn)":
                df_kn = load_and_filter([Path(p) for p in kn_by_date.get(d, [])], "kn", d)
                make_heatmap(df_kn, min_conf, f"KōreroNET • {d.isoformat()}")
            else:
                df_bn = load_and_filter([Path(p) for p in bn_by_date.get(d, [])], "bn", d)
                df_kn = load_and_filter([Path(p) for p in kn_by_date.get(d, [])], "kn", d)
                make_heatmap(pd.concat([df_bn, df_kn], ignore_index=True), min_conf, f"Combined (BN+KN) • {d.isoformat()}")

# ===== TAB 2 =====
with tab_verify:
    st.subheader("Verify recordings (snapshots in Backup/…)")
    list_only = st.checkbox("List only (no download)", value=False)
    max_csv_per_src = st.slider("Max CSVs per source (snapshot)", 10, 2000, 300, 10)
    colc1, colc2 = st.columns(2)
    with colc1:
        if st.button("Clear CSV cache"):
            shutil.rmtree(CSV_CACHE, ignore_errors=True)
            CSV_CACHE.mkdir(parents=True, exist_ok=True)
            st.success("CSV cache cleared.")
    with colc2:
        if st.button("Clear audio cache"):
            shutil.rmtree(CHUNK_CACHE, ignore_errors=True)
            CHUNK_CACHE.mkdir(parents=True, exist_ok=True)
            st.success("Audio cache cleared.")

    snaps, s_mode = list_snapshots_any()
    if not snaps:
        st.warning("No snapshot folders like YYYYMMDD_HHMMSS found under Backup/."); st.stop()

    snap_name = st.selectbox("Snapshot", options=[n for _, n, _ in snaps], index=0, key="verify_snap")
    snap_third = next(t for _, n, t in snaps if n == snap_name)  # drive folder ID or local path

    src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0, key="verify_src")
    min_conf_v = st.slider("Minimum confidence", 0.0, 1.0, 0.90, 0.01, key="verify_conf")

    ctx = {"snap": snap_name, "src": src_mode_v, "conf": float(min_conf_v), "cap": max_csv_per_src, "list": list_only}
    if st.session_state.get("v_ctx") != ctx:
        st.session_state["v_ctx"] = ctx
        st.session_state["v_loaded"] = False
        st.session_state["v_df"] = pd.DataFrame()

    def _gather_csvs_local(snapshot_path: str, src_mode: str) -> List[Tuple[str, str]]:
        todo = []
        if src_mode in ("KōreroNET (KN)", "Combined"):
            kn_dir = os.path.join(snapshot_path, "koreronet")
            for p in sorted(glob.glob(os.path.join(kn_dir, "*.csv")))[:max_csv_per_src]:
                todo.append(("kn", p))
        if src_mode in ("BirdNET (BN)", "Combined"):
            bn_dir = os.path.join(snapshot_path, "birdnet")
            for p in sorted(glob.glob(os.path.join(bn_dir, "*.csv")))[:max_csv_per_src]:
                todo.append(("bn", p))
        return todo

    def _gather_csvs_drive(snapshot_folder_id: str, src_mode: str) -> List[Tuple[str, Dict[str,Any], str]]:
        kids = list_children(snapshot_folder_id, max_items=2000)
        sub_kn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name")=="koreronet"]
        sub_bn = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name")=="birdnet"]
        todo = []
        if src_mode in ("KōreroNET (KN)", "Combined") and sub_kn:
            kn_id = sub_kn[0]["id"]
            kn_kids = [f for f in list_children(kn_id, max_items=2000) if f.get("name","").lower().endswith(".csv")]
            kn_kids.sort(key=lambda m: m.get("name",""))
            for m in kn_kids[:max_csv_per_src]:
                todo.append(("kn", m, kn_id))
        if src_mode in ("BirdNET (BN)", "Combined") and sub_bn:
            bn_id = sub_bn[0]["id"]
            bn_kids = [f for f in list_children(bn_id, max_items=2000) if f.get("name","").lower().endswith(".csv")]
            bn_kids.sort(key=lambda m: m.get("name",""))
            for m in bn_kids[:max_csv_per_src]:
                todo.append(("bn", m, bn_id))
        return todo

    def _load_with_progress_local(csv_list: List[Tuple[str, str]]) -> pd.DataFrame:
        if list_only:
            st.info(f"(List only) Local CSVs found: {len(csv_list)}"); return pd.DataFrame()
        prog = st.progress(0.0); status = st.empty()
        frames = []  # <<< FIXED
        n = max(1, len(csv_list))
        for i, (kind, path) in enumerate(csv_list, 1):
            status.text(f"Parsing {i}/{n}: {os.path.basename(path)}")
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
                out["ChunkPath"] = out["chunk_file"].apply(lambda f: os.path.join(os.path.dirname(path), f) if isinstance(f, str) else "")
                out["ChunkName"] = out["chunk_file"].apply(lambda f: os.path.basename(f) if isinstance(f, str) else "")
                out["ChunkDriveFolderId"] = ""
                frames.append(out)
            except Exception:
                pass
            prog.progress(i/n)
        status.text("Processing…"); prog.progress(1.0)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_with_progress_drive(csv_list: List[Tuple[str, Dict[str,Any], str]], snap_id: str) -> pd.DataFrame:
        if list_only:
            st.info(f"(List only) Drive CSVs found: {len(csv_list)}"); return pd.DataFrame()
        prog = st.progress(0.0); status = st.empty()
        frames = []  # <<< FIXED
        n = max(1, len(csv_list))
        for i, (kind, meta, chunk_folder_id) in enumerate(csv_list, 1):
            status.text(f"Downloading & parsing {i}/{n}: {meta.get('name')}")
            try:
                subdir = f"snap_{snap_id}/{ 'koreronet' if kind=='kn' else 'birdnet'}"
                csv_path = ensure_csv_cached_by_meta(meta, subdir=subdir)
                df = pd.read_csv(str(csv_path), sep=None, engine="python")
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
                out["ChunkName"] = out["chunk_file"].apply(lambda f: os.path.basename(f) if isinstance(f, str) else "")
                out["ChunkDriveFolderId"] = chunk_folder_id
                out["ChunkPath"] = ""
                frames.append(out)
            except Exception:
                pass
            prog.progress(i/n)
        status.text("Processing…"); prog.progress(1.0)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # LOAD
    if st.button("Load detections", type="primary"):
        with st.status("Gathering CSV list…", expanded=True) as s:
            if s_mode == "local":
                csv_list_local = _gather_csvs_local(snap_third, src_mode_v)
                s.update(label=f"Found {len(csv_list_local)} CSVs (local).")
                raw = _load_with_progress_local(csv_list_local)
            else:
                csv_list_drive = _gather_csvs_drive(snap_third, src_mode_v)
                s.update(label=f"Found {len(csv_list_drive)} CSVs (Drive).")
                raw = _load_with_progress_drive(csv_list_drive, snap_third)
        if list_only:
            st.info("List-only mode: no detections loaded. Disable to parse files.")
        else:
            std_v = raw
            def _actual_time(src_name: str, offset_seconds: float):
                m = re.match(r"^(\d{8})_(\d{6})", os.path.basename(str(src_name)))
                if not m: return pd.NaT
                base = datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")
                try:
                    return base + timedelta(seconds=float(offset_seconds or 0.0))
                except Exception:
                    return pd.NaT
            if not std_v.empty:
                use_off = std_v["det_start"].where(std_v["det_start"].notna(), std_v["start_s"])
                times = [_actual_time(s, o) for s, o in zip(std_v["src"], use_off)]
                std_v = pd.DataFrame({
                    "Label": std_v["label"].astype(str),
                    "Confidence": std_v["prob"].astype(float),
                    "ActualTime": times,
                    "Kind": std_v["Kind"],
                    "ChunkPath": std_v.get("ChunkPath", pd.Series([""]*len(std_v))),
                    "ChunkName": std_v.get("ChunkName", pd.Series([""]*len(std_v))),
                    "ChunkDriveFolderId": std_v.get("ChunkDriveFolderId", pd.Series([""]*len(std_v))),
                    "Start_s": std_v["start_s"],
                    "End_s": std_v["end_s"],
                }).dropna(subset=["ActualTime"]).sort_values("ActualTime").reset_index(drop=True)
                std_v = std_v[std_v["Confidence"] >= float(min_conf_v)]
            st.session_state["v_df"] = std_v
            st.session_state["v_loaded"] = True
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("v_idx::"):
                    del st.session_state[k]
            st.success("Detections loaded.")

    if not st.session_state.get("v_loaded", False) or list_only:
        st.info("Adjust settings and click **Load detections** (disable 'List only' to parse).")
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
        if st.button("⏮ Prev"): idx = (idx - 1) % len(playlist); autoplay = True
    with col2:
        if st.button("▶ Play"): autoplay = True
    with col3:
        if st.button("⏭ Next"): idx = (idx + 1) % len(playlist); autoplay = True
    st.session_state[idx_key] = idx
    with col4:
        st.markdown(f"**{species}** — {len(playlist)} detections on **{day_pick.isoformat()}** | index {idx+1}/{len(playlist)}")

    row = playlist.iloc[idx]
    meta = (f"**Confidence:** {row['Confidence']:.3f}  |  **Time:** {row['ActualTime']}  |  "
            f"Segment: {float(row['Start_s']):.2f}–{float(row['End_s']):.2f}s")
    st.markdown(meta)

    def _play_audio(row: pd.Series, auto: bool):
        data = None
        path_str = str(row.get("ChunkPath","") or "")
        if path_str and os.path.isfile(path_str):
            try:
                with open(path_str, "rb") as f:
                    data = f.read()
            except Exception:
                data = None
        if (data is None) and DRIVE_ENABLED:
            chunk_name = str(row.get("ChunkName","") or "")
            folder_id = str(row.get("ChunkDriveFolderId","") or "")
            if chunk_name and folder_id:
                subdir = f"{snap_name}/{row.get('Kind','UNK')}"
                cached = ensure_chunk_cached(chunk_name, folder_id, subdir=subdir)
                if cached and cached.exists():
                    try:
                        with open(cached, "rb") as f:
                            data = f.read()
                    except Exception:
                        data = None
        if data is None:
            st.warning("Cannot open audio chunk (not found or download failed)."); return
        if auto:
            import base64
            b64 = base64.b64encode(data).decode()
            st.markdown(f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>', unsafe_allow_html=True)
        else:
            st.audio(data, format="audio/wav")

    _play_audio(row, autoplay)

# ===== TAB 3 =====
with tab3:
    if not DRIVE_ENABLED:
        st.info("Drive not configured. Set GDRIVE_FOLDER_ID and service_account in Streamlit secrets.")
    else:
        st.caption("Top-level of your Google Drive Folder ID")
        max_items = st.slider("Max items to show", 50, 1000, 300, 50)
        with st.spinner("Listing…"):
            kids = list_children(GDRIVE_FOLDER_ID, max_items=max_items)
        bn = [k for k in kids if k.get("name","").lower().startswith("bn") and k.get("name","").lower().endswith(".csv")]
        kn = [k for k in kids if k.get("name","").lower().startswith("kn") and k.get("name","").lower().endswith(".csv")]
        st.info(f"bn*.csv: {len(bn)} | kn*.csv: {len(kn)}")
        st.dataframe(
            [{"name": k.get("name"), "id": k.get("id"), "mimeType": k.get("mimeType"),
              "modifiedTime": k.get("modifiedTime"), "size": k.get("size")} for k in kids],
            use_container_width=True, hide_index=True
        )
