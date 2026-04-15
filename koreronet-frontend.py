# KōreroNET Dashboard UI/UX Revision Skill File v2

## PROJECT CONTEXT
Bird acoustic monitoring dashboard built with Streamlit (Python). Monitors species detections from audio nodes. Sidebar navigation already implemented. Currently one node: Auckland-Sunnyhills. Sources: KōreroNET (kn), BirdNET (bn), Combined.

## PRIORITY BUG: ALL PAGE TITLES CLIPPED
Every page title has its top half cut off. This is caused by CSS margin/padding issue or st.markdown custom styling pushing titles under a fixed element.
### Fix:
```python
# Add top padding/margin to main content area
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    /* Remove any negative margins on h1/h2 */
    h1, h2, h3 {
        margin-top: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)
```

## SIDEBAR NAVIGATION: Remove Emojis
Emojis render inconsistently across OS/browser. Replace with clean text-only labels or minimal unicode symbols.
### Fix:
```python
with st.sidebar:
    st.markdown("### KōreroNET")
    st.caption("Bird Acoustic Monitoring")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Nodes", "Detections", "Verify", "Power", "Log", "Search"],
        label_visibility="collapsed"
    )
```
Remove the bird emoji from the sidebar title. Use plain text "KōreroNET" only. Remove "Bird Acoustic Monitoring" caption or make it slightly larger (14px, medium gray).

## SIDEBAR BRANDING
- Remove all emojis from branding (no bird emoji, no icons)
- "KōreroNET" in bold 20px
- "Bird Acoustic Monitoring" in 13px, color #9E9E9E (visible in both themes)
- No logo image unless a proper SVG/PNG is available
- Divider below branding, above navigation

## ABOUT SECTION: Remove Emojis
Replace emoji bullets with plain text or simple dashes.
### Before (bad):
```
🔊 Detects bird species from audio
🎧 Verify recordings
⚙️ Monitor node health
```
### After (clean):
```
— Detects bird species from field audio recordings
— Verify and validate species detections
— Monitor node health, power, and connectivity
```

## NODES TAB: Fix Metric Sizing
"Online" status text is comically oversized. Battery "77%" and Energy "0.1 Wh" are also too large.
### Fix:
```python
# Don't use st.metric for status — it makes text too large
# Use custom HTML with controlled sizing
st.markdown("**Status:** Online", help=None)
st.markdown("**Battery:** 77%")
st.markdown("**Energy:** 0.1 Wh")
st.markdown("**Last data:** 2026-04-15 00:00")
```
Or use st.metric but with smaller custom CSS:
```python
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)
```
Ensure "Last data" timestamp is NOT truncated. Use full date format: "15 Apr 2026, 00:00"

## DETECTIONS TAB: Fix Slider Clipping
Min confidence slider is clipped off-screen right. Filters are in columns that are too narrow.
### Fix:
```python
# Use wider column ratios
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    source = st.selectbox("Source", options)
with col2:
    day = st.date_input("Day")
with col3:
    confidence = st.slider("Min confidence", 0.0, 1.0, 0.5)
```
Or stack filters vertically instead of horizontal columns.

## DETECTIONS HEATMAP: Fix Background Color
The Peak activity heatmap has a CREAM/WHITE background that clashes with both dark and light themes.
### Fix:
```python
import plotly.express as px

# Use a colorscale that works on dark backgrounds
fig = px.imshow(
    heatmap_data,
    color_continuous_scale="Hot",  # or "Inferno", "YlOrRd"
    # Set transparent or dark background
)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E0E0E0",
)
# For light theme compatibility, detect theme and swap:
# paper_bgcolor="rgba(0,0,0,0)" works for both if chart bg is transparent
```
The heatmap cells with value 0 should be dark/transparent, NOT cream/white.

## LOG TAB: Remove Level Column
Level column shows ALL DASHES "—" because the log format doesn't contain severity levels. The column is completely useless.
### Fix:
```python
# REMOVE the Level column entirely
# REMOVE the severity filter multiselect
# Keep only: Timestamp | Message
# Parse timestamp FROM the message text: [2026-04-15 01:17:56]

import re

def parse_log(raw):
    rows = []
    for line in raw.strip().split("\n"):
        match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(.*)", line)
        if match:
            rows.append({"Timestamp": match.group(1), "Message": match.group(2)})
        else:
            rows.append({"Timestamp": "", "Message": line})
    return pd.DataFrame(rows)
```
Remove severity filter pill. Keep only the search box for filtering messages.

## LOG TAB: Fix Timestamp Duplication
Timestamps appear BOTH in the Timestamp column (empty) AND inside the Message column.
### Fix:
Extract timestamp from message and put it in the Timestamp column. Remove timestamp prefix from Message column so it's not duplicated.

## SEARCH TAB: Fix Available Species Display
"Available:" shows a massive comma-separated text dump that wraps and is overwhelming.
### Fix:
```python
# REMOVE the "Available:" text dump entirely
# Instead, use a multiselect with searchable dropdown
species_input = st.multiselect(
    "Select species",
    options=all_species_list,
    default=None,
    placeholder="Type to search species..."
)
```
This replaces both the text input AND the "Available:" dump with a single clean searchable multi-select.

## SEARCH TAB: Fix Source Dropdown Truncation
"Combi..." truncated because column is too narrow.
### Fix:
```python
# Widen the source column or use full labels
col1, col2 = st.columns([3, 1])
# Or use shorter labels:
source_options = {"Combined": "combined", "KōreroNET": "kn", "BirdNET": "bn"}
```

## SEARCH RESULTS: Fix Peak Table Clipping
"Avg/day" column is clipped on right edge.
### Fix:
```python
st.dataframe(peak_df, use_container_width=True)
```

## THEME-ADAPTIVE CSS
```python
# Inject CSS that works in both light and dark themes
st.markdown("""
<style>
    /* Fix title clipping - most critical */
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { margin-top: 0.5rem !important; }

    /* Metric sizing */
    [data-testid="stMetricValue"] { font-size: 24px; }

    /* Stats cards - auto-adapt */
    [data-testid="stMetricValue"] {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)
```

## FONT AND SIZING RULES
- Page titles: 24px bold (not larger — Streamlit h2)
- Section subtitles: 18px semibold
- Body text: 14px regular
- Stat numbers: 24px bold (NOT the default st.metric giant size)
- Stat labels: 12px, medium gray
- Sidebar nav items: 14px regular (no emojis)
- Sidebar branding: 20px bold title, 13px caption
- Tables: 13px, full container width
- Log messages: 13px monospace

## LAYOUT RULES
- All charts: use_container_width=True
- All tables: use_container_width=True
- Filter columns: minimum ratio [1,1,2] so sliders have room
- Node status: use st.markdown bold text, NOT st.metric (too large)
- Available species: use st.multiselect, NOT text dump
- Log: 2 columns only (Timestamp, Message), NO Level column

## COMPLETE BUG LIST
1. All page titles clipped at top (CSS padding issue)
2. Heatmap cream/white background clashes with dark theme
3. Sidebar emojis inconsistent — remove all emojis from nav
4. About section emojis look informal — use dashes or plain text
5. Node "Online" metric text oversized
6. Node "Last data" date truncated
7. Log Level column all dashes — remove entirely
8. Log timestamps duplicated (in column AND in message)
9. Search "Available:" species text dump — use multiselect
10. Source dropdown "Combi..." truncated
11. Min confidence slider clipped off-screen
12. Peak table "Avg/day" column clipped
13. Bird emoji in sidebar title looks informal
14. "Bird Acoustic Monitoring" caption barely visible
15. Heatmap 0-value cells should be transparent, not cream
