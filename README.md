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

**How nodes are found (online):** the dashboard discovers a node folder if it is
either (a) a child of the folder in `GDRIVE_FOLDER_ID`, **or** (b) any folder
named `From the node …` that the **service account can access**. Because the
node folders usually live at your **My Drive root** (a service account can't list
My Drive root), the reliable way is (b): **share each `From the node <N>` folder
with the service-account `client_email`** (Viewer). No parent folder or secret
change is required — share a new node's folder and it appears automatically.

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
GDRIVE_FOLDER_ID = "any_shared_node_folder_id"   # a parent of node folders, or
                                                 # just one node folder — nodes are
                                                 # also auto-discovered by name (b)

[service_account]                          # Google service account (drive.readonly)
type = "service_account"
project_id = "…"
private_key_id = "…"
private_key = "-----BEGIN PRIVATE KEY-----\n…\n-----END PRIVATE KEY-----\n"
client_email = "…@….iam.gserviceaccount.com"
# … remaining service-account fields …
```

**Share every `From the node <N>` folder with the service account's
`client_email`** (Viewer). That is what makes each node visible and its data
loadable. `GDRIVE_FOLDER_ID` can point at any shared node folder (or a parent);
additional nodes are discovered by name as long as they're shared with the SA.

### Offline / local mode

Set `OFFLINE_DEPLOY = True` in `koreronet-frontend.py` and point
`KORERONET_DATA_ROOT` (env var) at a local folder laid out exactly like the Drive
tree above. The same multi-node discovery works against the local filesystem.

## Run

```bash
pip install -r requirements.txt
streamlit run koreronet-frontend.py
```
