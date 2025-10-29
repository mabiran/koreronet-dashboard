#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Drive Diagnostics (Streamlit)

What it does
------------
• Reads GDRIVE_FOLDER_ID and either [service_account] (preferred) or SERVICE_ACCOUNT_JSON from st.secrets.
• Normalizes the private_key formatting automatically.
• Verifies Google API imports.
• Builds a Drive client and shows service account identity.
• Tests: drive.about(), folder metadata, children listing (paged).
• Provides a tree view (depth-limited) and BN/KN CSV quick-count.
• Produces a single JSON "Summary" you can copy/paste to me.

Secrets format (preferred)
--------------------------
GDRIVE_FOLDER_ID = "YOUR_FOLDER_ID"
[service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----
... (real newlines, no \n escapes) ...
-----END PRIVATE KEY-----"""
client_email = "....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/...."
universe_domain = "googleapis.com"
"""
import json
import traceback
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import Counter

import streamlit as st

# -------------------------------
# Page config & style
# -------------------------------
st.set_page_config(page_title="GDrive Diagnostics", layout="wide")
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
def safe_tb(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}\n{traceback.format_exc()}"

def normalize_private_key(pk: str) -> str:
    """Convert literal \\n to real newlines; ensure PEM header/footer line breaks."""
    if not isinstance(pk, str):
        return pk
    # Convert escaped newlines to actual newlines
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    # Ensure header/footer line breaks even if pasted as one line
    if "-----BEGIN PRIVATE KEY-----" in pk and "-----END PRIVATE KEY-----" in pk:
        # If the block is one line, insert breaks
        if "-----BEGIN PRIVATE KEY-----\n" not in pk:
            pk = pk.replace("-----BEGIN PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----\n")
        if "\n-----END PRIVATE KEY-----" not in pk:
            pk = pk.replace("-----END PRIVATE KEY-----", "\n-----END PRIVATE KEY-----")
    return pk

@dataclass
class Diag:
    has_GDRIVE_FOLDER_ID: bool = False
    has_service_account_table: bool = False
    has_SERVICE_ACCOUNT_JSON: bool = False
    google_api_import_ok: bool = False
    service_account_email: str = None
    project_id: str = None
    private_key_id_tail: str = None
    drive_client_ok: bool = False
    drive_client_error: str = None
    about_ok: bool = False
    about_error: str = None
    folder_ok: bool = False
    folder_error: str = None
    folder_mimeType: str = None
    folder_name: str = None
    children_ok: bool = False
    children_error: str = None
    total_children_returned: int = 0
    csv_bn_count: int = 0
    csv_kn_count: int = 0
    type_counts: Dict[str, int] = None

def render_header():
    st.title("🔧 Google Drive Diagnostics")
    st.caption("Verifies Streamlit secrets ➜ builds Drive client ➜ lists folder contents ➜ provides copyable summary.")

render_header()

# -------------------------------
# Secrets
# -------------------------------
GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID")
SA_TABLE = st.secrets.get("service_account")            # preferred: TOML table ➜ dict
SA_JSON  = st.secrets.get("SERVICE_ACCOUNT_JSON")       # legacy: JSON string

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
        st.warning("Missing `GDRIVE_FOLDER_ID` in Streamlit secrets.")
    if not (diag.has_service_account_table or diag.has_SERVICE_ACCOUNT_JSON):
        st.warning("Provide a service account via `[service_account]` table (recommended) or `SERVICE_ACCOUNT_JSON`.")

# Optional: allow overriding folder ID during testing
override_id = st.text_input("Test with this Folder ID (optional)", value=GDRIVE_FOLDER_ID or "", help="Leave empty to use the value from secrets.")
FOLDER_ID = override_id.strip() or (GDRIVE_FOLDER_ID or "")

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
    st.code(safe_tb(e), language="text")

with st.expander("Python dependency check", expanded=True):
    st.write({
        "google_api_import_ok": diag.google_api_import_ok
    })

if not diag.google_api_import_ok:
    st.stop()

# -------------------------------
# Build credentials & client
# -------------------------------
drive = None
if diag.has_service_account_table or diag.has_SERVICE_ACCOUNT_JSON:
    try:
        if SA_TABLE:
            sa_info = dict(SA_TABLE)
        else:
            sa_info = json.loads(SA_JSON)

        # Normalize private key formatting
        if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
            sa_info["private_key"] = normalize_private_key(sa_info["private_key"])

        diag.service_account_email = sa_info.get("client_email")
        diag.project_id = sa_info.get("project_id")
        pkid = sa_info.get("private_key_id")
        diag.private_key_id_tail = (pkid[-6:] if pkid else None)

        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        diag.drive_client_ok = True
    except Exception as e:
        diag.drive_client_ok = False
        diag.drive_client_error = safe_tb(e)

with st.expander("Service account / client build", expanded=True):
    st.write({
        "service_account_email": diag.service_account_email,
        "project_id": diag.project_id,
        "private_key_id_tail": diag.private_key_id_tail,
        "drive_client_ok": diag.drive_client_ok,
        "drive_client_error": diag.drive_client_error,
    })
    if not diag.drive_client_ok:
        st.stop()

# -------------------------------
# About test
# -------------------------------
try:
    about = drive.about().get(fields="user/displayName,user/emailAddress,storageQuota").execute()
    diag.about_ok = True
except Exception as e:
    diag.about_ok = False
    diag.about_error = safe_tb(e)
    about = None

with st.expander("About() test", expanded=True):
    st.write({"about_ok": diag.about_ok, "about_error": diag.about_error})
    if about:
        st.json(about)

# -------------------------------
# Folder metadata test
# -------------------------------
folder_meta = None
if not FOLDER_ID:
    st.error("No Folder ID provided (from secrets or override).")
    st.stop()

try:
    folder_meta = drive.files().get(
        fileId=FOLDER_ID,
        fields="id, name, mimeType, owners, permissions"
    ).execute()
    diag.folder_ok = True
    diag.folder_mimeType = folder_meta.get("mimeType")
    diag.folder_name = folder_meta.get("name")
except Exception as e:
    diag.folder_ok = False
    diag.folder_error = safe_tb(e)

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
        st.warning("The provided ID is not a folder. Listing children may not work as expected.")

# -------------------------------
# List children (paged)
# -------------------------------
def list_children(folder_id: str, max_items: int = 500) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    token = None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, md5Checksum)",
            pageSize=min(100, max_items - len(items)),
            pageToken=token,
            orderBy="folder,name_natural"
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token or len(items) >= max_items:
            break
    return items

children: List[Dict[str, Any]] = []
try:
    limit = st.slider("Max items to list", 50, 2000, 500, 50)
    with st.spinner("Listing folder children…"):
        children = list_children(FOLDER_ID, max_items=limit)
    diag.children_ok = True
    diag.total_children_returned = len(children)
except Exception as e:
    diag.children_ok = False
    diag.children_error = safe_tb(e)

with st.expander("Children listing", expanded=True):
    st.write({
        "children_ok": diag.children_ok,
        "children_error": diag.children_error,
        "total_children_returned": diag.total_children_returned,
    })
    if children:
        # quick type histogram
        type_hist = Counter([c.get("mimeType", "?") for c in children])
        diag.type_counts = dict(type_hist)
        st.write({"mimeType_counts": dict(type_hist)})

        # Show a concise table
        def row(c):
            return {
                "name": c.get("name"),
                "id": c.get("id"),
                "mimeType": c.get("mimeType"),
                "modifiedTime": c.get("modifiedTime"),
                "size": c.get("size"),
                "md5Checksum": c.get("md5Checksum"),
            }
        table = [row(c) for c in children]
        st.dataframe(table, use_container_width=True, hide_index=True)

        # Quick CSV counts
        bn = [c for c in children if c.get("name", "").lower().startswith("bn") and c.get("name", "").lower().endswith(".csv")]
        kn = [c for c in children if c.get("name", "").lower().startswith("kn") and c.get("name", "").lower().endswith(".csv")]
        diag.csv_bn_count = len(bn)
        diag.csv_kn_count = len(kn)
        st.info(f"bn*.csv: {len(bn)}  |  kn*.csv: {len(kn)}")

# -------------------------------
# Depth-limited tree (optional)
# -------------------------------
st.subheader("Folder tree (depth-limited)")
max_depth = st.slider("Max depth", 0, 5, 2)
max_per_folder = st.slider("Max items per folder", 5, 200, 50, 5)

def is_folder(item: Dict[str, Any]) -> bool:
    return item.get("mimeType") == "application/vnd.google-apps.folder"

@st.cache_data(show_spinner=False)
def fetch_children(folder_id: str, cap: int) -> List[Dict[str, Any]]:
    """Cached helper for recursion."""
    try:
        return list_children(folder_id, max_items=cap)
    except Exception:
        return []

def render_tree(folder_id: str, name: str, depth: int, max_depth: int, cap: int):
    indent = "  " * depth
    st.write(f"{indent}📁 {name} ({folder_id})")
    if depth >= max_depth:
        return
    kids = fetch_children(folder_id, cap)
    if not kids:
        return
    # folders first, then files
    folders = [k for k in kids if is_folder(k)]
    files = [k for k in kids if not is_folder(k)]
    for d in folders:
        render_tree(d["id"], d.get("name", "(no name)"), depth + 1, max_depth, cap)
    for f in files[: cap]:
        st.write(f"{'  '*(depth+1)}📄 {f.get('name')} [{f.get('mimeType')}] ({f.get('id')})")

if diag.folder_ok and diag.children_ok:
    render_tree(FOLDER_ID, diag.folder_name or "(root)", 0, max_depth, max_per_folder)

# -------------------------------
# Copyable JSON Summary
# -------------------------------
st.subheader("Summary (copy/paste this to ChatGPT if something fails)")
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
    "If **Children** is empty but client/folder are OK, check that the folder is shared with the "
    "service account email shown above (as Viewer), and that the files are indeed inside the exact Folder ID."
)
