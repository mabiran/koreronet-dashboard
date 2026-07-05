# KōreroNET Dashboard

Streamlit dashboard for KoreroNet acoustic-monitoring nodes. Reads each node's
Google Drive `From the node <N>` folder (detections, snapshots, power logs) and
renders detections, verification, power, logs, search, and a node map.

## Multi-node support

The dashboard supports **any number of nodes**. Point it at the **parent** Drive
folder that holds all the node folders:

```
<root>/                      ← GDRIVE_FOLDER_ID secret points HERE
├── From the node 0/
│   ├── node.ini             ← name / lat / long for this node
│   ├── Backup/ …
│   ├── Power logs/ …
│   └── *_master.csv …
├── From the node 1/
│   ├── node.ini
│   └── …
├── From the node 2/
│   ├── node.ini
│   └── …
└── To the node N/           ← ignored by the dashboard
```

Every `From the node <N>` folder becomes a selectable node in the sidebar, and
each one is plotted on the map from its own `.ini`.

### The node `.ini`

Drop a small `.ini` in each `From the node <N>` folder (any `*.ini`;
`node.ini` / `location.ini` / `site.ini` are preferred if several exist):

```ini
name: Auckland — Sunnyhills
lat:  -36.9003
long: 174.8839
```

- Keys are **case-insensitive**; `:` or `=` both work.
- `long`, `lon`, `lng`, `longitude` are all accepted; `lat`/`latitude` too.
- A node with no `.ini` (or no coordinates) still appears in the selector and its
  data loads — it just isn't drawn on the map.

If the configured folder has **no** `From the node <N>` sub-folders, the
dashboard treats it as a **single node** (backward compatible with the old
setup).

## Configuration (`.streamlit/secrets.toml`)

```toml
GDRIVE_FOLDER_ID = "your_root_folder_id"   # the PARENT of all node folders

[service_account]                          # Google service account (drive.readonly)
type = "service_account"
project_id = "…"
private_key_id = "…"
private_key = "-----BEGIN PRIVATE KEY-----\n…\n-----END PRIVATE KEY-----\n"
client_email = "…@….iam.gserviceaccount.com"
# … remaining service-account fields …
```

Share the parent Drive folder (and its sub-folders) with the service account's
`client_email` (Viewer is enough).

### Offline / local mode

Set `OFFLINE_DEPLOY = True` in `koreronet-frontend.py` and point
`KORERONET_DATA_ROOT` (env var) at a local folder laid out exactly like the Drive
tree above. The same multi-node discovery works against the local filesystem.

## Run

```bash
pip install -r requirements.txt
streamlit run koreronet-frontend.py
```
