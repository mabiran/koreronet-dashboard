# app.py — Google Drive Diagnostics Only

import os, json, traceback
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="GDrive Diagnostics", layout="wide")

st.title("Google Drive Diagnostics")

# --- Secrets we expect ---
GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID")
SERVICE_ACCOUNT_JSON = st.secrets.get("SERVICE_ACCOUNT_JSON")

with st.expander("Secrets status", expanded=True):
    st.write({
        "has_GDRIVE_FOLDER_ID": bool(GDRIVE_FOLDER_ID),
        "has_SERVICE_ACCOUNT_JSON": bool(SERVICE_ACCOUNT_JSON),
    })
    if not GDRIVE_FOLDER_ID:
        st.warning("Missing `GDRIVE_FOLDER_ID` in Streamlit secrets.")
    if not SERVICE_ACCOUNT_JSON:
        st.warning("Missing `SERVICE_ACCOUNT_JSON` in Streamlit secrets.")

# --- Try importing the Google API libs (if this fails, you need requirements.txt) ---
import_ok = True
import_err = None
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from googleapiclient.errors import HttpError
except Exception as e:
    import_ok = False
    import_err = f"{type(e).__name__}: {e}"

with st.expander("Python dependency check", expanded=True):
    st.write({"google_api_import_ok": import_ok})
    if not import_ok:
        st.error(
            "Google API libraries not installed. Add these to **requirements.txt**:\n\n"
            "```\n"
            "google-api-python-client\n"
            "google-auth\n"
            "google-auth-httplib2\n"
            "google-auth-oauthlib\n"
            "```"
        )
        st.stop()

# --- Build Drive client (if secrets exist) ---
drive = None
client_ok, client_err = False, None
sa_email = None
sa_project = None
sa_key_tail = None

if GDRIVE_FOLDER_ID and SERVICE_ACCOUNT_JSON:
    try:
        sa_info = SERVICE_ACCOUNT_JSON
        if isinstance(sa_info, str):
            sa_info = json.loads(sa_info)

        sa_email = sa_info.get("client_email")
        sa_project = sa_info.get("project_id")
        pkid = sa_info.get("private_key_id")
        sa_key_tail = (pkid[-6:] if pkid else None)

        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        client_ok = True
    except Exception as e:
        client_err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

with st.expander("Service account / client build", expanded=True):
    st.write({
        "service_account_email": sa_email,
        "project_id": sa_project,
        "private_key_id_tail": sa_key_tail,
        "drive_client_ok": client_ok,
        "drive_client_error": client_err,
    })
    if not client_ok:
        st.stop()

# --- UI inputs ---
st.subheader("Run checks")
folder_id = st.text_input(
    "Folder ID to test",
    value=(GDRIVE_FOLDER_ID or ""),
    help="Paste just the ID (the long string from the Drive URL after /folders/), not the whole URL."
)
max_children = st.number_input("Max children to list", min_value=1, max_value=2000, value=50)
do_tree = st.checkbox("List folder tree (Breadth-first, capped)", value=False)
max_tree_nodes = st.number_input("Tree cap", min_value=10, max_value=5000, value=500)
try_download = st.checkbox("Try downloading first CSV (bn*/kn*) to /tmp", value=False)

run = st.button("Run diagnostics", type="primary")

def diag_about(drive):
    try:
        about = drive.about().get(fields="user, kind").execute()
        return True, about, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

def diag_folder_get(drive, fid):
    try:
        meta = drive.files().get(
            fileId=fid,
            fields="id, name, mimeType, parents, driveId, permissions, capabilities",
            supportsAllDrives=True,
        ).execute()
        return True, meta, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

def diag_list_children(drive, fid, limit):
    try:
        resp = drive.files().list(
            q=f"'{fid}' in parents and trashed=false",
            fields="files(id, name, mimeType, size, md5Checksum, driveId)",
            pageSize=int(limit),
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        return True, resp.get("files", []), None
    except Exception as e:
        return False, [], f"{type(e).__name__}: {e}"

def diag_tree(drive, root_fid, cap):
    from collections import deque
    out = []
    q = deque([(root_fid, "")])  # (folder_id, path prefix)
    seen = set()
    nodes = 0
    try:
        # small helper to list children
        def _kids(fid):
            r = drive.files().list(
                q=f"'{fid}' in parents and trashed=false",
                fields="files(id, name, mimeType, size)",
                pageSize=200,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            return r.get("files", [])

        while q and nodes < cap:
            fid, prefix = q.popleft()
            if fid in seen:
                continue
            seen.add(fid)
            for it in _kids(fid):
                nodes += 1
                path = f"{prefix}/{it['name']}".lstrip("/")
                out.append({"path": path, **it})
                if it.get("mimeType", "").endswith("folder"):
                    q.append((it["id"], path))
                if nodes >= cap:
                    break
        return True, out, None
    except Exception as e:
        return False, out, f"{type(e).__name__}: {e}"

def try_download_first_csv(drive, fid):
    try:
        resp = drive.files().list(
            q=f"'{fid}' in parents and trashed=false and (name contains 'bn' or name contains 'kn') and name contains '.csv'",
            fields="files(id, name)",
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        # If none at root, we won't recurse here—this is just a quick download poke.
        if not files:
            return False, None, "No bn*/kn* CSVs found directly under this folder (not searching subfolders)."
        file_id = files[0]["id"]
        name = files[0]["name"]
        out_path = Path("/tmp") / name
        req = drive.files().get_media(fileId=file_id)
        with open(out_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return True, str(out_path), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

if run:
    summary = {}
    # Step A: about()
    ok, about, err = diag_about(drive)
    summary["about_ok"] = ok
    summary["about_user"] = about.get("user", {}) if ok else None
    summary["about_error"] = err

    # Step B: folder metadata
    ok, meta, err = diag_folder_get(drive, folder_id)
    summary["folder_get_ok"] = ok
    summary["folder_meta"] = meta if ok else None
    summary["folder_get_error"] = err

    # Step C: list children
    if ok:
        ok2, kids, err2 = diag_list_children(drive, folder_id, max_children)
        summary["list_children_ok"] = ok2
        summary["children_count"] = len(kids) if ok2 else 0
        summary["list_children_error"] = err2
        summary["children_sample"] = kids[:50] if ok2 else []
    else:
        summary["list_children_ok"] = False
        summary["list_children_error"] = "Skipped because folder_get failed."

    # Step D: optional tree
    if ok and do_tree:
        ok3, tree_items, err3 = diag_tree(drive, folder_id, max_tree_nodes)
        summary["tree_ok"] = ok3
        summary["tree_count"] = len(tree_items) if ok3 else 0
        summary["tree_error"] = err3
        # show first 200 paths
        summary["tree_sample_paths"] = [t["path"] for t in tree_items[:200]] if ok3 else []
    else:
        summary["tree_ok"] = None

    # Step E: optional download
    if ok and try_download:
        ok4, path, err4 = try_download_first_csv(drive, folder_id)
        summary["download_test_ok"] = ok4
        summary["download_path"] = path
        summary["download_error"] = err4
    else:
        summary["download_test_ok"] = None

    st.subheader("Summary")
    bullets = []
    bullets.append(f"- about(): {'✅' if summary['about_ok'] else '❌'}")
    bullets.append(f"- folder get: {'✅' if summary['folder_get_ok'] else '❌ ' + (summary['folder_get_error'] or '')}")
    if summary.get("children_sample") is not None:
        bullets.append(f"- children list: {'✅' if summary['list_children_ok'] else '❌'} "
                       f"({summary.get('children_count', 0)} items)")
    if summary.get("tree_ok") is True:
        bullets.append(f"- tree: ✅ {summary.get('tree_count', 0)} items (showing first 200 paths below)")
    if summary.get("download_test_ok") is True:
        bullets.append(f"- download test: ✅ saved to {summary.get('download_path')}")
    elif summary.get("download_test_ok") is False:
        bullets.append(f"- download test: ❌ {summary.get('download_error')}")
    st.markdown("\n".join(bullets))

    st.markdown("**Copy & paste this JSON to me:**")
    st.code(json.dumps(summary, indent=2), language="json")

    # Pretty viewers
    if summary.get("children_sample"):
        st.markdown("**Children (first 50)**")
        st.dataframe(summary["children_sample"])

    if summary.get("tree_ok") and summary.get("tree_sample_paths"):
        st.markdown("**Tree sample paths (first 200)**")
        st.code("\n".join(summary["tree_sample_paths"]), language="text")

else:
    st.info("Fill the Folder ID (defaults to your secret), choose options, then click **Run diagnostics**.")
    if sa_email:
        st.caption(
            "Make sure this service account has access to the folder (and Shared Drive if applicable): "
            f"**{sa_email}**"
        )
