"""EcoSort - Waste Classifier (YOLO11s-cls + Streamlit)."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

# key (model class name, lowercase): display name, recyclable, color, bin, tip, icon
CATEGORIES = {
    "plastic": {"name": "Plastic", "recyclable": True, "color": "#1E88E5",
                "bin": "Blue bin", "icon": "🧴",
                "tip": "Empty and rinse the container, remove the cap, then place it in the plastic bin."},
    "paper": {"name": "Paper / Cardboard", "recyclable": True, "color": "#F9A825",
              "bin": "Yellow bin", "icon": "📄",
              "tip": "Keep it clean and dry. Flatten cardboard boxes before recycling."},
    "glass": {"name": "Glass", "recyclable": True, "color": "#00897B",
              "bin": "Green bin", "icon": "🍾",
              "tip": "Rinse it and remove the lid. Be careful with broken glass."},
    "metal": {"name": "Metal", "recyclable": True, "color": "#546E7A",
              "bin": "Grey bin", "icon": "🥫",
              "tip": "Empty the can and rinse it. Crush it if possible to save space."},
    "trash": {"name": "Non-recyclable", "recyclable": False, "color": "#E53935",
              "bin": "Black bin", "icon": "🗑️",
              "tip": "This item can't be recycled. Place it in the general waste bin."},
}
UNKNOWN = {"name": "Unknown", "recyclable": False, "color": "#757575",
           "bin": "-", "icon": "❓", "tip": ""}

st.set_page_config(page_title="EcoSort | Waste Classifier", page_icon="♻️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
  padding: 28px 34px; border-radius: 18px; color: #fff; margin-bottom: 26px;
  box-shadow: 0 6px 18px rgba(46,125,50,.25);}
.hero h1 {margin: 0; font-size: 2.2rem; color: #fff; padding: 0;}
.hero p {margin: 6px 0 0 0; font-size: 1.05rem; opacity: .95; color: #fff;}
.result-card {background: #fff; color: #1b2a1f; border-radius: 16px; padding: 22px 24px;
  box-shadow: 0 4px 16px rgba(0,0,0,.10); border-left: 10px solid var(--c);}
.result-card .icon {font-size: 2.6rem; line-height: 1;}
.result-card .title {font-size: 2rem; font-weight: 800; color: var(--c); margin: 4px 0;}
.badge {display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 700;
  font-size: .9rem; color: #fff; margin: 4px 8px 4px 0;}
.badge.ok {background: #2E7D32;} .badge.no {background: #C62828;}
.badge.bin {background: var(--c);}
.conf-wrap {background: #E8EEE9; border-radius: 999px; height: 14px; margin: 10px 0 4px 0;}
.conf-fill {height: 14px; border-radius: 999px; background: var(--c);}
.tip {background: #F1F8F2; color: #1b2a1f; border-radius: 10px; padding: 12px 14px; margin-top: 12px;}
.warn-card {background: #FFF8E1; color: #5d4037; border-left: 10px solid #FFA000;
  border-radius: 16px; padding: 22px 24px; box-shadow: 0 4px 16px rgba(0,0,0,.08);}
.prob-row {display: flex; align-items: center; margin: 7px 0; font-size: .92rem;}
.prob-name {width: 135px; font-weight: 600;}
.prob-bar {flex: 1; background: #E8EEE9; border-radius: 8px; height: 16px; margin: 0 10px;}
.prob-fill {height: 16px; border-radius: 8px;}
.prob-val {width: 52px; text-align: right; font-weight: 600;}
.legend-item {display: flex; align-items: center; padding: 9px 12px; margin: 6px 0;
  border-radius: 10px; background: #fff; color: #1b2a1f; border-left: 6px solid var(--c);
  box-shadow: 0 1px 4px rgba(0,0,0,.08);}
.legend-item .nm {font-weight: 700; flex: 1; margin-left: 8px;}
.legend-item .rc {font-size: .75rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; color:#fff;}
.section-title {font-size: 1.3rem; font-weight: 700; margin: 26px 0 8px 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the AI model...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def classify(model: YOLO, image: Image.Image):
    result = model.predict(image, verbose=False)[0]
    probs = {result.names[i].lower(): float(p) for i, p in enumerate(result.probs.data.tolist())}
    label = max(probs, key=probs.get)
    return label, probs[label], probs


def info(label: str) -> dict:
    return CATEGORIES.get(label, {**UNKNOWN, "name": label.title()})


def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        mode = st.radio("Image source", ["📁 Upload image", "📷 Use camera"])
        threshold = st.slider("Minimum confidence", 0.30, 0.95, 0.60, 0.05,
                              help="Below this value the app will answer 'Not sure'.")
        st.markdown("## 🗂️ Waste classes")
        legend = ""
        for c in CATEGORIES.values():
            badge_bg = "#2E7D32" if c["recyclable"] else "#C62828"
            badge = "Recyclable" if c["recyclable"] else "Not recyclable"
            legend += (f'<div class="legend-item" style="--c:{c["color"]}">'
                       f'<span>{c["icon"]}</span><span class="nm">{c["name"]}</span>'
                       f'<span class="rc" style="background:{badge_bg}">{badge}</span></div>')
        st.markdown(legend, unsafe_allow_html=True)
        st.write("")
        if st.button("🧹 Clear history", use_container_width=True):
            st.session_state.history = []
    return mode, threshold


def show_result(label: str, conf: float, threshold: float):
    if conf < threshold:
        st.markdown(
            f'<div class="warn-card"><div class="icon" style="font-size:2.4rem">🤔</div>'
            f'<div style="font-size:1.6rem;font-weight:800">Not sure</div>'
            f'<p>The model is only {conf:.0%} confident (best guess: <b>{info(label)["name"]}</b>). '
            f'Try a clearer photo, better lighting, or a single item in the frame.</p></div>',
            unsafe_allow_html=True)
        return
    c = info(label)
    badge = '<span class="badge ok">♻️ Recyclable</span>' if c["recyclable"] \
        else '<span class="badge no">🚫 Not recyclable</span>'
    st.markdown(
        f'<div class="result-card" style="--c:{c["color"]}">'
        f'<div class="icon">{c["icon"]}</div>'
        f'<div class="title">{c["name"]}</div>'
        f'{badge}<span class="badge bin">🗑️ {c["bin"]}</span>'
        f'<div style="margin-top:10px;font-weight:600">Confidence: {conf:.1%}</div>'
        f'<div class="conf-wrap"><div class="conf-fill" style="width:{conf*100:.1f}%"></div></div>'
        f'<div class="tip">💡 <b>How to dispose:</b> {c["tip"]}</div></div>',
        unsafe_allow_html=True)


def show_probs(probs: dict):
    html = '<div class="section-title" style="margin-top:22px">📊 Probabilities</div>'
    for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        c = info(k)
        html += (f'<div class="prob-row"><div class="prob-name">{c["icon"]} {c["name"]}</div>'
                 f'<div class="prob-bar"><div class="prob-fill" '
                 f'style="width:{v*100:.1f}%;background:{c["color"]}"></div></div>'
                 f'<div class="prob-val">{v:.0%}</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.markdown(
        '<div class="hero"><h1>♻️ EcoSort – Smart Waste Classifier</h1>'
        '<p>Upload a photo or use your camera, and the AI will tell you what the waste is '
        'and which bin it belongs in.</p></div>', unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Copy your trained `best.pt` into the `models/` folder and restart the app.")
        st.stop()

    model = load_model(str(MODEL_PATH))
    if "history" not in st.session_state:
        st.session_state.history = []

    mode, threshold = sidebar()

    if mode.endswith("Upload image"):
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Take a photo")

    if file is not None:
        image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        label, conf, probs = classify(model, image)

        left, right = st.columns(2, gap="large")
        with left:
            st.image(image, use_container_width=True, caption="Your image")
        with right:
            show_result(label, conf, threshold)
            show_probs(probs)

        shown = info(label)["name"] if conf >= threshold else "Not sure"
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": getattr(file, "name", "camera"),
            "Result": shown,
            "Confidence": f"{conf:.1%}",
        })

    if st.session_state.history:
        st.markdown('<div class="section-title">🕘 History</div>', unsafe_allow_html=True)
        hist = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False).encode("utf-8-sig"),
                           "history.csv", "text/csv")


main()
