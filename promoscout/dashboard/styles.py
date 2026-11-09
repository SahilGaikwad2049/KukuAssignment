"""Local, responsive visual system for the PromoScout Streamlit workspace."""

CSS = r"""
<style>
:root {
  --ps-red: #EC1C24; --ps-red-soft: #D9534F; --ps-bg: #000107;
  --ps-surface: #15151B; --ps-raised: #202027; --ps-text: #F5F5F5;
  --ps-muted: #A5A5AD; --ps-border: #303039; --ps-green: #66C99A;
}
html, body, [data-testid="stAppViewContainer"], button, input, textarea, select {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", Ubuntu, Cantarell, sans-serif !important;
  font-feature-settings: "kern" 1, "liga" 1;
  text-rendering: optimizeLegibility;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--ps-bg); color: var(--ps-text); }
[data-testid="stHeader"] { background: var(--ps-bg); }
[data-testid="stMain"] { min-width: 0; }
.block-container { max-width: 1440px; padding-top: 4rem; padding-bottom: 4rem; }
h1, h2, h3, p, label, span, div { overflow-wrap: anywhere; }
h1, h2, h3 { letter-spacing: -.02em; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size:.9rem; }
[data-testid="stSidebar"] { background: #09090D; border-right: 1px solid var(--ps-border); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--ps-muted); }
[data-testid="stMetric"] { background: var(--ps-surface); border: 1px solid var(--ps-border); padding: 1rem 1.1rem; border-radius: 14px; min-height: 112px; }
[data-testid="stMetricLabel"] { color: var(--ps-muted) !important; font-size: .86rem; }
[data-testid="stMetricValue"] { color: var(--ps-text) !important; font-weight: 700; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--ps-border) !important; border-radius: 16px !important; background: rgba(21,21,27,.64); }
[data-testid="stExpander"] { border-color: var(--ps-border); border-radius: 12px; background: #111116; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stMultiSelect"] [data-baseweb="select"] { background: var(--ps-raised); }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button { min-height: 42px; border-radius: 10px; border: 1px solid var(--ps-border); transition: border-color .16s ease, transform .16s ease, background .16s ease; }
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover { border-color: var(--ps-red-soft); transform: translateY(-1px); }
[data-testid="stButton"] button[kind="primary"] { background: var(--ps-red); border-color: var(--ps-red); color: white; }
[data-testid="stButton"] button[kind="primary"]:disabled,
[data-testid="stButton"] button[kind="primary"][aria-disabled="true"] {
  background:#34343b !important; border-color:#45454e !important; color:#9999a2 !important;
  opacity:.78; cursor:not-allowed; transform:none !important; box-shadow:none !important;
}
[data-testid="stButton"] button[kind="primary"]:disabled:hover,
[data-testid="stButton"] button[kind="primary"][aria-disabled="true"]:hover {
  background:#34343b !important; border-color:#45454e !important; transform:none !important;
}
.ps-brand { display:flex; align-items:center; gap:.7rem; margin:.3rem 0 1.6rem; }
.ps-demo-banner { max-width:1440px; margin:0 auto .85rem; padding:.8rem 1rem; border:1px solid #a83a40; border-radius:12px; color:#fff; background:linear-gradient(90deg,#4a1418,#251014); font-size:.92rem; line-height:1.45; }
.ps-mark { display:inline-grid; place-items:center; width:34px; height:34px; border-radius:10px; background:var(--ps-red); color:#fff; font-weight:900; letter-spacing:-.08em; }
.ps-wordmark { font-weight:800; letter-spacing:-.04em; color:var(--ps-text); font-size:1.12rem; }
.ps-wordmark small { display:block; color:var(--ps-muted); letter-spacing:.12em; text-transform:uppercase; font-size:.58rem; font-weight:600; margin-top:1px; }
.ps-eyebrow { display:flex; align-items:center; gap:.55rem; color:#FF777B; text-transform:uppercase; letter-spacing:.14em; font-size:.8rem; line-height:1.35; font-weight:800; margin:0 0 .55rem; text-shadow:0 0 16px rgba(236,28,36,.2); }
.ps-eyebrow::before { content:""; display:inline-block; width:18px; height:2px; flex:0 0 18px; border-radius:2px; background:var(--ps-red); }
.ps-page-title { font-size:clamp(1.9rem,3.4vw,2.8rem); line-height:1.12; font-weight:720; margin:.05rem 0 .55rem; }
.ps-subtitle { color:var(--ps-muted); font-size:1rem; max-width:760px; line-height:1.55; margin:0 0 1.4rem; }
.ps-metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:1rem 0 1.5rem; }
.ps-metric { background:var(--ps-surface); border:1px solid var(--ps-border); border-radius:14px; padding:15px 17px; min-width:0; }
.ps-metric-label { color:var(--ps-muted); font-size:.82rem; line-height:1.35; }
.ps-metric-value { color:var(--ps-text); font-weight:800; font-size:clamp(1.45rem,2.5vw,2rem); line-height:1.2; margin:.25rem 0; }
.ps-metric-note { color:#85858e; font-size:.74rem; }
.ps-status { display:inline-flex; align-items:center; gap:.45rem; border:1px solid var(--ps-border); border-radius:99px; padding:.32rem .65rem; color:var(--ps-muted); font-size:.76rem; white-space:nowrap; }
.ps-dot { width:7px; height:7px; border-radius:50%; background:#777; }
.ps-status.done .ps-dot { background:var(--ps-green); box-shadow:0 0 10px #66c99a66; }
.ps-status.attention .ps-dot { background:#E1A24A; }
.ps-status.active .ps-dot { background:var(--ps-red); }
.ps-pipeline { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; }
.ps-pipeline-step { padding:14px; border:1px solid var(--ps-border); background:var(--ps-surface); border-radius:12px; min-width:0; }
.ps-pipeline-step b { display:block; font-size:.88rem; margin-bottom:.5rem; }
.ps-pipeline-step small { color:var(--ps-muted); line-height:1.4; }
.ps-pipeline-compact { display:flex; flex-wrap:wrap; gap:.55rem; margin:.35rem 0 1.1rem; }
.ps-pipeline-compact .ps-status { background:#111116; }
.ps-candidate { border:1px solid var(--ps-border); background:linear-gradient(145deg,#17171e,#111116); border-radius:16px; padding:18px; margin:.45rem 0 .9rem; }
.ps-candidate-top { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap; }
.ps-candidate-id { color:var(--ps-muted); font-size:.78rem; letter-spacing:.04em; }
.ps-candidate-title { font-size:1.08rem; font-weight:750; margin:.18rem 0; }
.ps-tag { display:inline-block; border:1px solid var(--ps-border); border-radius:99px; color:var(--ps-muted); font-size:.72rem; padding:.22rem .55rem; margin:.1rem .2rem .15rem 0; }
.ps-tag-red { color:#ff777b; border-color:#76252a; background:#311316; }
.ps-meta { color:var(--ps-muted); font-size:.83rem; line-height:1.5; }
.ps-hook { border-left:2px solid var(--ps-red); padding-left:.8rem; margin:.8rem 0; }
.ps-ending { border-left:2px solid #62626c; padding-left:.8rem; margin:.8rem 0; }
.ps-event { display:grid; grid-template-columns:116px minmax(0,1fr); gap:12px; padding:13px 14px; border:1px solid var(--ps-border); border-radius:12px; background:#111116; margin:.45rem 0; }
.ps-event-time { font-variant-numeric:tabular-nums; color:#ff777b; font-size:.82rem; }
.ps-event-content { min-width:0; color:var(--ps-text); line-height:1.5; white-space:pre-wrap; }
.ps-empty { padding:22px; border:1px dashed #40404a; border-radius:14px; background:#101015; }
.ps-empty h3 { margin:0 0 .35rem; font-size:1rem; }
.ps-empty p { color:var(--ps-muted); margin:0; }
.ps-divider { height:1px; background:var(--ps-border); margin:1.3rem 0; }
@media(max-width:1100px) {
  .ps-metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .ps-pipeline { grid-template-columns:repeat(3,minmax(0,1fr)); }
}
@media(max-width:900px) {
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.7rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex:1 1 46% !important; width:auto !important; min-width:0 !important; }
}
@media(max-width:760px) {
  .block-container { padding:3.5rem 1rem 3rem; }
  .ps-demo-banner { margin-left:1rem; margin-right:1rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex:1 1 100% !important; width:100% !important; }
  .ps-page-title { font-size:1.9rem; }
  .ps-subtitle { font-size:.92rem; margin-bottom:1.1rem; }
  .ps-metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
  .ps-metric { padding:12px; }
  .ps-metric-value { font-size:1.45rem; }
  .ps-pipeline { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
  .ps-pipeline-step { padding:11px; }
  .ps-event { grid-template-columns:1fr; gap:5px; padding:12px; }
  .ps-event-time { font-size:.76rem; }
  [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { min-height:46px; }
}
@media(max-width:390px) {
  .ps-metric-grid { grid-template-columns:1fr 1fr; }
  .ps-pipeline { grid-template-columns:1fr; }
  .block-container { padding-left:.75rem; padding-right:.75rem; }
}
</style>
"""


def inject_styles(st) -> None:
    st.markdown(CSS, unsafe_allow_html=True)

