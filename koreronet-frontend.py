#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDrive Diagnostics — Streamlit (service account via st.secrets)

What this page checks:
1) Presence & shape of secrets (GDRIVE_FOLDER_ID, [service_account] table, optional SERVICE_ACCOUNT_JSON).
2) Normalization of service-account private_key (handles \n vs real newlines, forces header/footer line breaks).
3) Builds googleapiclient Drive v3 client with drive.readonly scope.
4) Calls drive.about(), fetches folder metadata, and lists children (paged).
5) Shows depth-limited "tree view".
6) Emits a copyable JSON Summary with non-sensitive fields only.

Requirements (already in your requirements.txt):
- google-api-python-client
- google-auth
- google-auth-httplib2
"""

import json
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

st.set_page_config(page_title="🔧 GDrive Diagnostics", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top:1rem; padding-bottom:1rem;}
      code, pre {font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Helpers
# -------------------------------
def tb(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

def normalize_private_key(pk: str) -> str:
    """Convert literal \\n to real newlines; ensure BEGIN/END have line breaks."""
    if not isinstance(pk, str):
        return pk
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    # Ensure proper line breaks for header/footer even if pasted as one line
    if "-----BEGIN PRIVATE KEY-----" in pk and "-----END PRIVATE KEY-----" in pk:
        if "-----BEGIN PRIVATE KEY-----\n" not in pk:
            pk = pk.replace("-----BEGIN PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----\n", 1)
        if "\n-----END PRIVATE KEY-----" not in pk:
            pk = pk.replace("-----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----", 1)
    return pk

def safe_tail(s: Optional[str], n: int = 6) -> Optional[str]:
    if not s or not isinstance(s, str):
        return None
    return s[-n:] if len(s) >= n else s

# -------------------------------
# Dataclass for results
# -------------------------------
@dataclass
class Diag:
    # Secrets presence
    has_GDRIVE_FOLDER_ID: bool = False
    has_service_account_table: bool = False
    has_SERVICE_ACCOUNT_JSON: bool = False
    # Imports
    google_api_import_ok: bool = False
    # Client identity
    service_account_email: Optional[str] = None
    project_id: Optional[str] = None
    private_key_id_tail: Optional[str] = None
    private_key_len: Optional[int] = None
    private_key_has_begin: Optional[bool] = None
    private_key_has_end: Optional[bool] = None
    # Client build
    drive_client_ok: bool = False
    drive_client_error: Optional[str] = None
    # About
    about_ok: bool = False
    about_error: Optional[str] = None
    # Folder
    folder_ok: bool = False
    folder_error: Optional[str] = None
    folder_name: Optional[str] = None
    folder_mimeType: Optional[str] = None
    # Children
    children_ok: bool = False
    children_error: Optional[str] = None
    total_children_returned: int = 0
    type_counts: Optional[Dict[str, int]] = None
    csv_bn_count: int = 0
    csv_kn_count: int = 0

st.title("🔧 Google Drive Diagnostics")
st.caption("Checks Streamlit secrets → builds Drive client → inspects folder → produces a copyable JSON summary.")

# -------------------------------
# Read secrets (TOML table preferred)
# -------------------------------
GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID")
SA_TABLE = st.secrets.get("service_account")      # preferred
SA_JSON = st.secrets.get("SERVICE_ACCOUNT_JSON")  # optional legacy

diag = Diag(
    has_GDRIVE_FOLDER_ID=bool(GDRIVE_FOLDER_ID),
    has_service_account_table=bool(SA_TABLE),
    has_SERVICE_ACCOUNT_JSON=bool(SA_JSON),
)

with st.expander("Secrets status", expanded=True):
    st.write({
        "has_GDRIVE_FOLDER_ID": diag.has_GDRIVE_FOLDER_ID,
        "has_service_account_table": diag.has_service_account_table,
        "has_SERVICE_ACCOUNT_JSON": diag.has_SERVICE_ACCOUNT_JSON,
    })
    if not diag.has_GDRIVE_FOLDER_ID:
        st.error("Missing `GDRIVE_FOLDER_ID` in Streamlit secrets.")
    if not (diag.has_service_account_table or diag.has_SERVICE_ACCOUNT_JSON):
        st.error("Missing service account. Add a `[service_account]` table (recommended) or `SERVICE_ACCOUNT_JSON`.")

# Allow a one-off override for folder ID
FOLDER_ID = st.text_input(
    "Test Folder ID (optional override)",
    value=GDRIVE_FOLDER_ID or "",
    help="Leave empty to use GDRIVE_FOLDER_ID from secrets.",
).strip() or (GDRIVE_FOLDER_ID or "")

# -------------------------------
# Import Google API deps
# -------------------------------
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    diag.google_api_import_ok = True
except Exception as e:
    diag.google_api_import_ok = False
    st.error("Failed to import Google API client libraries.")
    st.code(tb(e), language="text")

with st.expander("Python dependency check", expanded=True):
    st.write({"google_api_import_ok": diag.google_api_import_ok})

if not diag.google_api_import_ok:
    st.stop()

# -------------------------------
# Build credentials & Drive client
# -------------------------------
drive = None
if not (diag.has_service_account_table or diag.has_SERVICE_ACCOUNT_JSON):
    st.stop()

try:
    if SA_TABLE:
        sa_info = dict(SA_TABLE)
    else:
        # JSON string case
        sa_info = json.loads(SA_JSON)

    # Normalize private key
    if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
        sa_info["private_key"] = normalize_private_key(sa_info["private_key"])
        diag.private_key_len = len(sa_info["private_key"])
        diag.private_key_has_begin = "-----BEGIN PRIVATE KEY-----" in sa_info["private_key"]
        diag.private_key_has_end = "-----END PRIVATE KEY-----" in sa_info["private_key"]

    diag.service_account_email = sa_info.get("client_email")
    diag.project_id = sa_info.get("project_id")
    diag.private_key_id_tail = safe_tail(sa_info.get("private_key_id"))

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

    # Build Drive v3 client
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    diag.drive_client_ok = True
except Exception as e:
    diag.drive_client_ok = False
    diag.drive_client_error = tb(e)

with st.expander("Service account / client build", expanded=True):
    st.write({
        "service_account_email": diag.service_account_email,
        "project_id": diag.project_id,
        "private_key_id_tail": diag.private_key_id_tail,
        "private_key_len": diag.private_key_len,
        "private_key_has_begin": diag.private_key_has_begin,
        "private_key_has_end": diag.private_key_has_end,
        "drive_client_ok": diag.drive_client_ok,
        "drive_client_error": diag.drive_client_error,
    })
if not diag.drive_client_ok:
    st.stop()

# -------------------------------
# About() test
# -------------------------------
about = None
try:
    about = drive.about().get(fields="user/displayName,user/emailAddress,storageQuota").execute()
    diag.about_ok = True
except Exception as e:
    diag.about_ok = False
    diag.about_error = tb(e)

with st.expander("About() test", expanded=True):
    st.write({"about_ok": diag.about_ok, "about_error": diag.about_error})
    if about:
        st.json(about)

# -------------------------------
# Folder metadata
# -------------------------------
if not FOLDER_ID:
    st.error("No Folder ID available (secrets or override).")
    st.stop()

folder_meta = None
try:
    folder_meta = drive.files().get(
        fileId=FOLDER_ID,
        fields="id,name,mimeType,driveId,owners,permissions,shared",
        supportsAllDrives=True,
    ).execute()
    diag.folder_ok = True
    diag.folder_name = folder_meta.get("name")
    diag.folder_mimeType = folder_meta.get("mimeType")
except HttpError as e:
    diag.folder_ok = False
    diag.folder_error = tb(e)
except Exception as e:
    diag.folder_ok = False
    diag.folder_error = tb(e)

with st.expander("Folder metadata", expanded=True):
    st.write({
        "folder_ok": diag.folder_ok,
        "folder_error": diag.folder_error,
        "folder_name": diag.folder_name,
        "folder_mimeType": diag.folder_mimeType,
    })
    if folder_meta:
        st.json(folder_meta)
    if diag.folder_ok and diag.folder_mimeType != "application/vnd.google-apps.folder":
        st.warning("The provided ID is not a folder. Children listing may not behave as expected.")

if not diag.folder_ok:
    st.stop()

# -------------------------------
# List children
# -------------------------------
def list_children(folder_id: str, max_items: int = 500) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    token = None
    while True:
        page_size = min(100, max_items - len(items))
        if page_size <= 0:
            break
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,md5Checksum,driveId)",
            pageSize=page_size,
            pageToken=token,
            orderBy="folder,name_natural",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="user",  # change to "drive" with driveId if you know it's a Shared Drive
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return items

children: List[Dict[str, Any]] = []
try:
    limit = st.slider("Max items to list", min_value=50, max_value=2000, value=500, step=50)
    with st.spinner("Listing folder children…"):
        children = list_children(FOLDER_ID, max_items=limit)
    diag.children_ok = True
    diag.total_children_returned = len(children)
except Exception as e:
    diag.children_ok = False
    diag.children_error = tb(e)

with st.expander("Children listing", expanded=True):
    st.write({
        "children_ok": diag.children_ok,
        "children_error": diag.children_error,
        "total_children_returned": diag.total_children_returned,
    })
    if children:
        type_hist = Counter(c.get("mimeType", "?") for c in children)
        diag.type_counts = dict(type_hist)
        st.write({"mimeType_counts": diag.type_counts})

        def to_row(c: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "name": c.get("name"),
                "id": c.get("id"),
                "mimeType": c.get("mimeType"),
                "modifiedTime": c.get("modifiedTime"),
                "size": c.get("size"),
                "md5Checksum": c.get("md5Checksum"),
            }

        st.dataframe([to_row(c) for c in children], use_container_width=True, hide_index=True)

        bn = [c for c in children if c.get("name", "").lower().startswith("bn") and c.get("name", "").lower().endswith(".csv")]
        kn = [c for c in children if c.get("name", "").lower().startswith("kn") and c.get("name", "").lower().endswith(".csv")]
        diag.csv_bn_count = len(bn)
        diag.csv_kn_count = len(kn)
        st.info(f"bn*.csv: {len(bn)}  |  kn*.csv: {len(kn)}")

# -------------------------------
# Folder tree (depth-limited)
# -------------------------------
st.subheader("Folder tree (depth-limited)")
max_depth = st.slider("Max depth", 0, 5, 2)
max_per_folder = st.slider("Max items per folder", 5, 200, 50, 5)

def is_folder(item: Dict[str, Any]) -> bool:
    return item.get("mimeType") == "application/vnd.google-apps.folder"

@st.cache_data(show_spinner=False)
def cached_children(folder_id: str, cap: int) -> List[Dict[str, Any]]:
    try:
        return list_children(folder_id, max_items=cap)
    except Exception:
        return []

def render_tree(folder_id: str, name: str, depth: int, max_depth: int, cap: int):
    st.write(f"{'  '*depth}📁 {name} ({folder_id})")
    if depth >= max_depth:
        return
    kids = cached_children(folder_id, cap)
    if not kids:
        return
    folders = [k for k in kids if is_folder(k)]
    files = [k for k in kids if not is_folder(k)]
    for d in folders:
        render_tree(d["id"], d.get("name", "(no name)"), depth + 1, max_depth, cap)
    for f in files[:cap]:
        st.write(f"{'  '*(depth+1)}📄 {f.get('name')} [{f.get('mimeType')}] ({f.get('id')})")

if diag.folder_ok and diag.children_ok:
    render_tree(FOLDER_ID, diag.folder_name or "(root)", 0, max_depth, max_per_folder)

# -------------------------------
# Copyable JSON Summary
# -------------------------------
st.subheader("Summary (copy/paste this if something fails)")
summary = {
    "Secrets": {
        "has_GDRIVE_FOLDER_ID": diag.has_GDRIVE_FOLDER_ID,
        "has_service_account_table": diag.has_service_account_table,
        "has_SERVICE_ACCOUNT_JSON": diag.has_SERVICE_ACCOUNT_JSON,
    },
    "Imports": {"google_api_import_ok": diag.google_api_import_ok},
    "Client": {
        "service_account_email": diag.service_account_email,
        "project_id": diag.project_id,
        "private_key_id_tail": diag.private_key_id_tail,
        "private_key_len": diag.private_key_len,
        "private_key_has_begin": diag.private_key_has_begin,
        "private_key_has_end": diag.private_key_has_end,
        "drive_client_ok": diag.drive_client_ok,
        "drive_client_error": diag.drive_client_error,
    },
    "About": {"about_ok": diag.about_ok, "about_error": diag.about_error},
    "Folder": {
        "folder_ok": diag.folder_ok,
        "folder_error": diag.folder_error,
        "folder_name": diag.folder_name,
        "folder_mimeType": diag.folder_mimeType,
    },
    "Children": {
        "children_ok": diag.children_ok,
        "children_error": diag.children_error,
        "total_children_returned": diag.total_children_returned,
        "csv_bn_count": diag.csv_bn_count,
        "csv_kn_count": diag.csv_kn_count,
        "mimeType_counts": diag.type_counts,
    },
}
st.code(json.dumps(summary, indent=2), language="json")

st.info(
    "If **Client** is OK but **Folder** fails: the ID may be wrong or not shared with the service account email above.\n"
    "If **Folder** is OK but **Children** is empty: ensure files are actually in that folder and not in a different Drive.\n"
    "If this is a **Shared Drive**, the service account must have at least Viewer access to that Shared Drive."
)
