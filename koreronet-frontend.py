#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KōreroNET Dashboard — Drive-backed (simplified)
- Tab 1: Heatmaps from bn*/kn* CSVs at Drive root (same logic you had).
- Tab 2: Verify with a proper calendar; dates come from *inside each CSV* (ActualTime),
         scanning all Backup/<snapshot>/{koreronet,birdnet}. No snapshot/slider clutter.
- Tab 3: Quick Drive browser.

Requires Streamlit secrets:
GDRIVE_FOLDER_ID = "..."
[service_account]  (or SERVICE_ACCOUNT_JSON)
"""
import os, io, re, glob, json, shutil
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- UI & global cache dirs ----------------
st.set_page_config(page_title="KōreroNET Dashboard", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:1rem;}
.stTabs [role="tablist"] {gap:.5rem;}
.stTabs [role="tab"] {padding:.6rem 1rem; border-radius:999px; border:1px solid #3a3a3a;}
</style>
""", unsafe_allow_html=True)
st.title("KōreroNET • Daily Dashboard")

CACHE_ROOT   = Path("/tmp/koreronet_cache")
CSV_CACHE    = CACHE_ROOT / "csv"
CHUNK_CACHE  = CACHE_ROOT / "chunks"
for p in (CSV_CACHE, CHUNK_CACHE):
    p.mkdir(parents=True, exist_ok=True)

# --------------- Drive Client ---------------
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
            st.error("Missing service account secrets.")
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

# --------------- Drive Helpers ---------------
def list_children(folder_id: str, max_items: int = 2000) -> List[Dict[str, Any]]:
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

def ensure_csv_cached(meta: Dict[str, Any], subdir: str) -> Path:
    local_path = (CSV_CACHE / subdir / meta["name"])
    if not local_path.exists():
        download_to(local_path, meta["id"])
    return local_path

def ensure_chunk_cached(chunk_name: str, folder_id: str, subdir: str) -> Optional[Path]:
    local_path = CHUNK_CACHE / subdir / chunk_name
    if local_path.exists():
        return local_path
    # search the folder for this file name
    for k in list_children(folder_id, max_items=2000):
        if k.get("name") == chunk_name:
            try:
                download_to(local_path, k["id"])
                return local_path
            except Exception:
                return None
    return None

# --------------- Local fallback (Tab 1 only) ---------------
DEFAULT_ROOT          = r"G:\My Drive\From the node"
BACKUP_ROOT_DEFAULT   = r"G:\My Drive\From the node\Backup"
ROOT_LOCAL            = os.getenv("KORERONET_DATA_ROOT", DEFAULT_ROOT)

@st.cache_data(show_spinner=False)
def list_csvs_local(root: str) -> Tuple[List[str], List[str]]:
    bn_paths = sorted(glob.glob(os.path.join(root, "bn*.csv")))
    kn_paths = sorted(glob.glob(os.path.join(root, "kn*.csv")))
    return bn_paths, kn_paths

# --------------- Root CSVs (Tab 1) ---------------
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

# --------------- Standardize + Heatmap ---------------
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

# --------------- Snapshot discovery (Drive) ---------------
SNAP_RE = re.compile(r"^(\d{8})_(\d{6})$")

@st.cache_data(show_spinner=False)
def list_snapshots_drive(root_folder_id: str) -> List[Dict[str, Any]]:
    backup = find_child_folder_by_name(root_folder_id, "Backup")
    if not backup: return []
    kids = list_children(backup["id"], max_items=2000)
    snaps = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and SNAP_RE.match(k.get("name",""))]
    snaps.sort(key=lambda m: m.get("name",""), reverse=True)
    return snaps  # each has id, name

def _list_source_csvs_drive(snapshot_id: str, source_folder_name: str) -> Tuple[str, List[Dict[str,Any]]]:
    """Returns (chunk_folder_id, csv_meta_list) for koreronet or birdnet under a snapshot."""
    kids = list_children(snapshot_id, max_items=2000)
    sub = [k for k in kids if k.get("mimeType")=="application/vnd.google-apps.folder" and k.get("name")==source_folder_name]
    if not sub: return "", []
    src_id = sub[0]["id"]
    files = [f for f in list_children(src_id, max_items=2000) if f.get("name","").lower().endswith(".csv")]
    files.sort(key=lambda m: m.get("name",""))
    return src_id, files

# --------------- Robust CSV parsers for Verify ---------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower().strip().replace(" ","").replace("_",""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip().replace(" ","").replace("_","")
        if key in low: return low[key]
    return None

def _std_from_detection_csv(df: pd.DataFrame, kind: str, chunk_folder_id: str) -> pd.DataFrame:
    """Return standardized rows. Prefer ActualTime from inside the CSV."""
    df = df.copy()
    # Label column
    label_col = _find_col(df, ["Label","Common name","common_name","species","class"])
    if not label_col:
        df["Label"] = "Unknown"
    else:
        df["Label"] = df[label_col].astype(str)

    # Confidence column
    conf_col = _find_col(df, ["Confidence","prob","score"])
    df["Confidence"] = pd.to_numeric(df[conf_col], errors="coerce") if conf_col else np.nan

    # Chunk filename
    file_col = _find_col(df, ["file","chunk_file","chunk","wav","chunkname"])
    df["ChunkName"] = df[file_col].astype(str) if file_col else ""

    # SRC + offsets (for fallback time calc)
    src_col   = _find_col(df, ["src","source","orig","origin","file_src"])
    start_col = _find_col(df, ["det_start","start_s","start","offset_s","offset"])
    df["_src"]  = df[src_col] if src_col else ""
    df["_off"]  = pd.to_numeric(df[start_col], errors="coerce") if start_col else np.nan

    # PRIMARY: ActualTime from inside CSV
    at_col = _find_col(df, ["ActualTime","Actual Time","actualtime"])
    if at_col:
        at = pd.to_datetime(df[at_col], errors="coerce", dayfirst=True)
    else:
        # FALLBACK: derive from src filename + offset
        def _from_src(s: Any, off: Any):
            m = re.match(r"^(\d{8})_(\d{6})", os.path.basename(str(s)))
            if not m: return pd.NaT
            base = datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")
            try: return base + timedelta(seconds=float(off or 0.0))
            except Exception: return pd.NaT
        at = pd.to_datetime([_from_src(s,o) for s,o in zip(df["_src"], df["_off"])])

    out = pd.DataFrame({
        "Label": df["Label"],
        "Confidence": df["Confidence"],
        "ActualTime": at,
        "Kind": kind.upper(),
        "ChunkName": df["ChunkName"],
        "ChunkDriveFolderId": chunk_folder_id,
    })
    out = out.dropna(subset=["ActualTime"])
    return out

# --------------- UI Tabs ---------------
tab1, tab_verify, tab3 = st.tabs(["📊 Detections", "🎧 Verify recordings", "📁 Drive"])

# ===== TAB 1: same logic, Drive-backed root CSVs =====
with tab1:
    st.caption("Root of Drive folder: uses bn*.csv / kn*.csv (dates parsed from ActualTime in those files).")
    if DRIVE_ENABLED:
        with st.status("Listing root CSVs…", expanded=False) as s:
            bn_meta, kn_meta = list_csvs_drive_root(GDRIVE_FOLDER_ID)
            s.update(label=f"Found BN={len(bn_meta)} KN={len(kn_meta)} at root")

        if not bn_meta and not kn_meta:
            # fallback to local (handy for offline)
            bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
            if not bn_local and not kn_local:
                st.error("No bn*/kn* CSVs at Drive root (or local fallback)."); st.stop()
            bn_paths = [Path(p) for p in bn_local]
            kn_paths = [Path(p) for p in kn_local]
        else:
            with st.expander("Download progress (root)", expanded=True):
                bn_paths = [ensure_csv_cached(m, subdir="root/bn") for m in bn_meta]
                kn_paths = [ensure_csv_cached(m, subdir="root/kn") for m in kn_meta]
    else:
        bn_local, kn_local = list_csvs_local(ROOT_LOCAL)
        if not bn_local and not kn_local:
            st.error("Drive not configured and no local fallback found."); st.stop()
        bn_paths = [Path(p) for p in bn_local]
        kn_paths = [Path(p) for p in kn_local]

    with st.status("Building date index…", expanded=False):
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
        options, help_txt = bn_dates, "Dates present in any BN file."
    else:
        options, help_txt = kn_dates, "Dates present in any KN file."

    if not options:
        st.warning(f"No available dates for {src}."); st.stop()

    d_default = options[-1]
    d = st.date_input("Day", value=d_default, min_value=options[0], max_value=options[-1], help=help_txt)
    if d not in options:
        earlier = [x for x in options if x <= d]
        d = earlier[-1] if earlier else options[0]
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
                make_heatmap(load_and_filter(bn_by_date.get(d, []), "bn", d), min_conf, f"BirdNET • {d.isoformat()}")
            elif src == "KōreroNET (kn)":
                make_heatmap(load_and_filter(kn_by_date.get(d, []), "kn", d), min_conf, f"KōreroNET • {d.isoformat()}")
            else:
                df_bn = load_and_filter(bn_by_date.get(d, []), "bn", d)
                df_kn = load_and_filter(kn_by_date.get(d, []), "kn", d)
                make_heatmap(pd.concat([df_bn, df_kn], ignore_index=True), min_conf, f"Combined (BN+KN) • {d.isoformat()}")

# ===== TAB 2: Verify — calendar + ActualTime *from CSVs inside Backup* =====
with tab_verify:
    st.caption("Scans **Backup/<snapshot>/{koreronet,birdnet}** on Drive. Dates are read from **inside each CSV**.")

    if not DRIVE_ENABLED:
        st.error("Drive not configured in secrets."); st.stop()

    # Minimal controls
    colA, colB = st.columns([2,1])
    with colA:
        src_mode_v = st.selectbox("Source", ["KōreroNET (KN)", "BirdNET (BN)", "Combined"], index=0)
    with colB:
        min_conf_v = st.slider("Min confidence", 0.0, 1.0, 0.90, 0.01)

    # Calendar (not constrained; we’ll report if no detections)
    day_pick = st.date_input("Day to verify", value=date.today())

    # Load button
    if st.button("Load day", type="primary"):
        snaps = list_snapshots_drive(GDRIVE_FOLDER_ID)
        if not snaps:
            st.warning("No Backup snapshots found on Drive."); st.stop()

        # Prepare CSV lists across ALL snapshots for the chosen source(s)
        src_sets: List[Tuple[str, List[Dict[str,Any]]]] = []
        with st.status("Listing snapshot CSVs…", expanded=False) as sst:
            if src_mode_v in ("KōreroNET (KN)", "Combined"):
                all_kn: List[Dict[str,Any]] = []
                kn_chunk_folder_ids: List[str] = []
                for sn in snaps:
                    kn_id, kn_files = _list_source_csvs_drive(sn["id"], "koreronet")
                    if kn_id and kn_files:
                        # bundle as tuples with parent folder id kept in meta
                        for m in kn_files:
                            m["_chunk_folder_id"] = kn_id
                            m["_snap_id"] = sn["id"]
                        all_kn.extend(kn_files)
                src_sets.append(("kn", all_kn))
            if src_mode_v in ("BirdNET (BN)", "Combined"):
                all_bn: List[Dict[str,Any]] = []
                for sn in snaps:
                    bn_id, bn_files = _list_source_csvs_drive(sn["id"], "birdnet")
                    if bn_id and bn_files:
                        for m in bn_files:
                            m["_chunk_folder_id"] = bn_id
                            m["_snap_id"] = sn["id"]
                        all_bn.extend(bn_files)
                src_sets.append(("bn", all_bn))
            total_files = sum(len(v) for _, v in src_sets)
            sst.update(label=f"Found {total_files} CSVs across snapshots.")

        # Download + parse only what’s needed; filter by chosen day while reading
        prog = st.progress(0.0)
        status = st.empty()
        frames: List[pd.DataFrame] = []
        done, seen = 0, max(1, total_files)

        for kind, metas in src_sets:
            for m in metas:
                done += 1
                status.text(f"Parsing {done}/{total_files}: {m.get('name')}")
                try:
                    subdir = f"snap_{m['_snap_id']}/{'koreronet' if kind=='kn' else 'birdnet'}"
                    csv_path = ensure_csv_cached(m, subdir=subdir)
                    # Read quickly; we only need a few columns to filter by date
                    df = pd.read_csv(str(csv_path), engine="python")
                    std = _std_from_detection_csv(df, kind, m["_chunk_folder_id"])
                    # Filter by date from inside CSV
                    std = std[std["ActualTime"].dt.date == day_pick]
                    if not std.empty:
                        # Apply confidence filter now to avoid large memory
                        std = std[pd.to_numeric(std["Confidence"], errors="coerce") >= float(min_conf_v)]
                        if not std.empty:
                            frames.append(std)
                except Exception:
                    pass
                prog.progress(done/seen)

        status.text("Merging…")
        prog.progress(1.0)
        if not frames:
            st.warning("No detections for that day from the CSV contents."); st.stop()

        std_v = pd.concat(frames, ignore_index=True).sort_values("ActualTime").reset_index(drop=True)
        st.session_state["v_df"] = std_v
        st.session_state["v_loaded_at"] = f"{day_pick.isoformat()}::{src_mode_v}::{min_conf_v}"

    # If not loaded yet, stop here
    std_v = st.session_state.get("v_df", pd.DataFrame())
    if std_v.empty:
        st.info("Pick a date and click **Load day**."); st.stop()

    # Playlist UI
    day_loaded_key = st.session_state.get("v_loaded_at", "")
    avail_days = sorted(std_v["ActualTime"].dt.date.unique())
    if len(avail_days) == 1 and avail_days[0] != day_pick:
        st.info(f"(Loaded day is {avail_days[0].isoformat()} based on CSV content.)")

    counts = std_v.groupby("Label").size().sort_values(ascending=False)
    species = st.selectbox(
        "Species",
        options=list(counts.index),
        format_func=lambda s: f"{s} — {counts[s]} detections",
        index=0,
        key=f"verify_species::{day_loaded_key}",
    )

    playlist = std_v[std_v["Label"] == species].sort_values("ActualTime").reset_index(drop=True)
    idx_key = f"v_idx::{day_loaded_key}::{species}"
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
        st.markdown(f"**{species}** — {len(playlist)} detections on **{avail_days[0].isoformat()}** | index {idx+1}/{len(playlist)}")

    row = playlist.iloc[idx]
    st.markdown(f"**Confidence:** {float(row['Confidence']):.3f}  |  **Time:** {row['ActualTime']}")

    # On-demand audio fetch (Drive)
    def _play_audio(row: pd.Series, auto: bool):
        chunk_name = str(row.get("ChunkName","") or "")
        folder_id  = str(row.get("ChunkDriveFolderId","") or "")
        if not (chunk_name and folder_id):
            st.warning("No chunk mapping in CSV."); return
        subdir = f"{row.get('Kind','UNK')}"
        cached = ensure_chunk_cached(chunk_name, folder_id, subdir=subdir)
        if not cached or not cached.exists():
            st.warning("Audio chunk not found in snapshot folder."); return
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

# ===== TAB 3: Quick Drive view =====
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
