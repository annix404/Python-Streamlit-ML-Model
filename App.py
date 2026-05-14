import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Live Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    .stApp {
        background: #0a0e1a;
        color: #00ff88;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        color: #00ff88 !important;
    }

    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 900;
        color: #00ff88;
        text-align: center;
        text-shadow: 0 0 20px #00ff8866;
        letter-spacing: 4px;
        padding: 20px 0;
        border-bottom: 1px solid #00ff8833;
        margin-bottom: 20px;
    }

    .metric-box {
        background: #111827;
        border: 1px solid #00ff8833;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-family: 'Share Tech Mono', monospace;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00ff88;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .alert-box {
        background: #ff003322;
        border: 1px solid #ff0033;
        border-radius: 8px;
        padding: 10px 16px;
        color: #ff4466;
        font-family: 'Share Tech Mono', monospace;
        margin: 5px 0;
    }

    .stButton > button {
        background: #00ff8822;
        color: #00ff88;
        border: 1px solid #00ff88;
        font-family: 'Orbitron', monospace;
        letter-spacing: 2px;
        border-radius: 4px;
        padding: 8px 24px;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #00ff8844;
        box-shadow: 0 0 12px #00ff8855;
    }

    .sidebar .stButton > button {
        width: 100%;
    }

    div[data-testid="stCheckbox"] label {
        color: #00ff88;
        font-family: 'Share Tech Mono', monospace;
    }

    .stSlider > div > div {
        background: #00ff88;
    }

    .saved-notice {
        background: #00ff8822;
        border: 1px solid #00ff88;
        border-radius: 6px;
        padding: 8px 14px;
        color: #00ff88;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">⬡ LIVE OBJECT DETECTION & TRACKING</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ CONTROLS")
    run = st.checkbox("▶ Enable Camera", value=False)
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.markdown("### 🚨 ALERTS")
    alert_objects = st.multiselect(
        "Alert on detecting:",
        ["person", "cell phone", "bottle", "laptop", "car", "dog", "cat", "backpack", "chair"],
        default=["person"]
    )
    
    st.markdown("---")
    st.markdown("### 💾 SAVE FRAMES")
    save_frames = st.checkbox("Auto-save detections", value=False)
    
    st.markdown("---")
    st.markdown("### 📋 LEGEND")
    st.markdown("""
    <div style='font-family: monospace; font-size: 0.8rem; color: #888;'>
    🟢 Tracked Object<br>
    🔴 Alert Object<br>
    📦 Bounding Box<br>
    🔢 Object ID
    </div>
    """, unsafe_allow_html=True)

# Layout columns
col_feed, col_stats = st.columns([3, 1])

with col_feed:
    st.markdown("### 📷 CAMERA FEED")
    frame_placeholder = st.empty()

with col_stats:
    st.markdown("### 📊 STATS")
    fps_placeholder = st.empty()
    count_placeholder = st.empty()
    objects_placeholder = st.empty()
    alert_placeholder = st.empty()

# Saved frames display
saved_notice = st.empty()

# Detection log
log_placeholder = st.empty()

# State
if "saved_count" not in st.session_state:
    st.session_state.saved_count = 0
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []

os.makedirs("saved_frames", exist_ok=True)

# Color palette for tracking IDs
COLORS = [
    (0, 255, 136), (0, 200, 255), (255, 200, 0),
    (255, 100, 200), (100, 200, 255), (200, 255, 100),
]

def get_color(track_id):
    return COLORS[int(track_id) % len(COLORS)]

def draw_box(frame, x1, y1, x2, y2, label, color, conf):
    # Draw filled top bar
    cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Label text
    cv2.putText(frame, f"{label} {conf:.0%}", (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Main loop
if run:
    cap = cv2.VideoCapture(0)
    track_history = defaultdict(list)
    prev_time = time.time()
    frame_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Cannot access webcam.")
            break

        frame_count += 1
        results = model.track(frame, persist=True, conf=confidence, verbose=False)

        object_counts = defaultdict(int)
        active_alerts = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, cid, conf_val in zip(boxes, track_ids, class_ids, confs):
                x1, y1, x2, y2 = box
                label = model.names[cid]
                object_counts[label] += 1

                color = get_color(tid)
                is_alert = label in alert_objects

                if is_alert:
                    color = (0, 0, 255)  # Red for alerts (BGR)
                    if label not in active_alerts:
                        active_alerts.append(label)

                draw_box(frame, x1, y1, x2, y2, f"#{tid} {label}", color, conf_val)

                # Track path
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > 20:
                    track_history[tid].pop(0)

                pts = track_history[tid]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(frame, pts[i - 1], pts[i], c, 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # Overlay FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)

        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Save frame
        if save_frames and frame_count % 30 == 0 and object_counts:
            fname = f"saved_frames/frame_{datetime.now().strftime('%H%M%S_%f')}.jpg"
            cv2.imwrite(fname, frame)
            st.session_state.saved_count += 1
            saved_notice.markdown(
                f'<div class="saved-notice">💾 Saved {st.session_state.saved_count} frames → saved_frames/</div>',
                unsafe_allow_html=True
            )

        # Stats panel
        total_objects = sum(object_counts.values())

        fps_placeholder.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{fps:.0f}</div>
            <div class="metric-label">FPS</div>
        </div>
        """, unsafe_allow_html=True)

        count_placeholder.markdown(f"""
        <div class="metric-box" style="margin-top:10px">
            <div class="metric-value">{total_objects}</div>
            <div class="metric-label">Objects</div>
        </div>
        """, unsafe_allow_html=True)

        # Object list
        obj_html = '<div style="margin-top:10px">'
        for obj, cnt in sorted(object_counts.items(), key=lambda x: -x[1]):
            obj_html += f'<div class="metric-box" style="margin-top:6px; display:flex; justify-content:space-between; padding:8px 12px;"><span style="color:#aaa;font-family:monospace;font-size:0.85rem">{obj}</span><span style="color:#00ff88;font-family:monospace;font-weight:bold">{cnt}</span></div>'
        obj_html += '</div>'
        objects_placeholder.markdown(obj_html, unsafe_allow_html=True)

        # Alerts
        if active_alerts:
            alert_html = ""
            for a in active_alerts:
                alert_html += f'<div class="alert-box">🚨 ALERT: {a.upper()} DETECTED</div>'
            alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
        else:
            alert_placeholder.empty()

    cap.release()

else:
    frame_placeholder.markdown("""
    <div style="
        background: #111827;
        border: 1px dashed #00ff8844;
        border-radius: 12px;
        height: 400px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #00ff8866;
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        letter-spacing: 3px;
    ">
        <div style="font-size:3rem;margin-bottom:16px">📷</div>
        <div>CAMERA OFFLINE</div>
        <div style="font-size:0.7rem;margin-top:8px;color:#444">Enable camera in sidebar to start</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family: monospace; color: #333; font-size: 0.75rem; padding: 10px;">
    YOLOv8 · Streamlit · OpenCV · Real-Time Object Detection & Tracking
</div>
""", unsafe_allow_html=True)