import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import streamlit as st
import tensorflow as tf
# pyrefly: ignore [missing-import]
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

TEST_DIR = BASE_DIR / "data" / "processed" / "test"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


model = load_model()
class_names = load_class_names()


# ============================================================
# CUSTOM CSS
#
# IMPORTANT:
# CSS is inside st.markdown.
# Visible HTML is rendered using st.html below.
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(34, 197, 94, 0.10),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 25%,
                rgba(16, 185, 129, 0.08),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #020806 0%,
                #06150d 45%,
                #020806 100%
            );

        color: #f5f7f6;
    }


    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }


    /* Hide Streamlit branding */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ========================================================
       NAVIGATION
    ======================================================== */

    .navbar {
        width: 100%;

        display: flex;
        justify-content: space-between;
        align-items: center;

        padding: 12px 18px;

        margin-bottom: 45px;

        border-radius: 18px;

        background: rgba(8, 24, 16, 0.72);

        border: 1px solid rgba(255,255,255,0.07);

        backdrop-filter: blur(18px);

        box-shadow:
            0 15px 50px rgba(0,0,0,0.25);
    }


    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }


    .brand-icon {
        width: 44px;
        height: 44px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 13px;

        background:
            linear-gradient(
                135deg,
                #22c55e,
                #16a34a
            );

        font-size: 22px;

        box-shadow:
            0 0 25px rgba(34,197,94,0.30);
    }


    .brand-name {
        font-size: 18px;
        font-weight: 800;
    }


    .brand-subtitle {
        color: #81948a;
        font-size: 11px;
        margin-top: 2px;
    }


    html {
        scroll-behavior: smooth;
    }


    .nav-links {
        display: flex;
        gap: 8px;
        align-items: center;
    }


    .nav-item {
        padding: 9px 15px;

        border-radius: 12px;

        color: #9ca9a2;

        font-size: 13px;

        text-decoration: none;

        display: inline-block;

        transition: all 0.2s ease;

        cursor: pointer;

        border: 1px solid transparent;
    }


    .nav-item:hover {
        color: #86efac;

        background: rgba(34,197,94,0.12);

        border: 1px solid rgba(34,197,94,0.22);
    }


    .nav-active {
        color: #86efac;

        background:
            rgba(34,197,94,0.15);

        border:
            1px solid rgba(34,197,94,0.18);
    }


    /* ========================================================
       FALLING LEAVES
    ======================================================== */

    .falling-leaves {
        position: fixed;

        top: 0;
        left: 0;

        width: 100%;
        height: 100%;

        pointer-events: none;

        overflow: hidden;

        z-index: 0;
    }


    .falling-leaf {
        position: absolute;

        top: -80px;

        opacity: 0.30;

        font-size: 24px;

        animation:
            leafFall linear infinite;
    }


    .leaf-1 {
        left: 5%;
        animation-duration: 14s;
        animation-delay: 0s;
    }

    .leaf-2 {
        left: 15%;
        animation-duration: 18s;
        animation-delay: 3s;
    }

    .leaf-3 {
        left: 28%;
        animation-duration: 15s;
        animation-delay: 7s;
    }

    .leaf-4 {
        left: 43%;
        animation-duration: 19s;
        animation-delay: 2s;
    }

    .leaf-5 {
        left: 58%;
        animation-duration: 16s;
        animation-delay: 6s;
    }

    .leaf-6 {
        left: 73%;
        animation-duration: 20s;
        animation-delay: 1s;
    }

    .leaf-7 {
        left: 87%;
        animation-duration: 15s;
        animation-delay: 8s;
    }

    .leaf-8 {
        left: 96%;
        animation-duration: 18s;
        animation-delay: 4s;
    }


    @keyframes leafFall {

        0% {
            transform:
                translate3d(0, -80px, 0)
                rotate(0deg);
        }

        25% {
            transform:
                translate3d(55px, 25vh, 0)
                rotate(90deg);
        }

        50% {
            transform:
                translate3d(-45px, 50vh, 0)
                rotate(180deg);
        }

        75% {
            transform:
                translate3d(60px, 75vh, 0)
                rotate(270deg);
        }

        100% {
            transform:
                translate3d(-20px, 110vh, 0)
                rotate(360deg);
        }
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero {
        position: relative;

        z-index: 2;

        padding: 15px 0 20px;
    }


    .hero-badge {
        display: inline-block;

        padding: 8px 14px;

        border-radius: 30px;

        color: #86efac;

        background:
            rgba(34,197,94,0.10);

        border:
            1px solid rgba(34,197,94,0.22);

        font-size: 12px;

        font-weight: 700;

        letter-spacing: 0.5px;

        margin-bottom: 18px;
    }


    .hero-title {
        font-size: 57px;

        line-height: 1.02;

        font-weight: 850;

        letter-spacing: -2px;

        margin: 0;
    }


    .hero-highlight {
        color: #4ade80;

        text-shadow:
            0 0 30px rgba(74,222,128,0.22);
    }


    .hero-description {
        color: #9aaba2;

        font-size: 16px;

        line-height: 1.7;

        max-width: 530px;

        margin-top: 18px;
    }


    /* ========================================================
       FEATURES
    ======================================================== */

    .feature {
        display: flex;

        align-items: center;

        gap: 14px;

        margin-top: 21px;
    }


    .feature-icon {
        width: 42px;
        height: 42px;

        display: flex;

        align-items: center;
        justify-content: center;

        border-radius: 13px;

        background:
            rgba(34,197,94,0.10);

        border:
            1px solid rgba(34,197,94,0.16);

        font-size: 18px;
    }


    .feature-title {
        font-size: 14px;

        font-weight: 700;
    }


    .feature-text {
        color: #7f9188;

        font-size: 12px;

        margin-top: 3px;
    }


    /* ========================================================
       GLASS CARD
    ======================================================== */

    .glass-card {
        position: relative;

        z-index: 2;

        background:
            linear-gradient(
                135deg,
                rgba(19,38,28,0.78),
                rgba(5,18,11,0.72)
            );

        border:
            1px solid rgba(255,255,255,0.08);

        border-radius: 22px;

        backdrop-filter: blur(18px);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.25);
    }


    /* ========================================================
       UPLOAD CARD
    ======================================================== */

    .upload-card {
        padding: 25px;

        min-height: 385px;
    }


    .upload-title {
        font-size: 20px;

        font-weight: 800;

        margin-bottom: 5px;
    }


    .upload-description {
        color: #81938a;

        font-size: 13px;

        margin-bottom: 22px;
    }


    .upload-zone {
        border:
            1px dashed rgba(74,222,128,0.35);

        border-radius: 18px;

        padding: 38px 15px;

        text-align: center;

        background:
            rgba(34,197,94,0.035);

        margin-bottom: 18px;
    }


    .upload-icon {
        font-size: 46px;

        margin-bottom: 10px;
    }


    .upload-main {
        font-size: 16px;

        font-weight: 700;
    }


    .upload-small {
        color: #71847a;

        font-size: 12px;

        margin-top: 7px;
    }


    /* ========================================================
       STREAMLIT FILE UPLOADER
    ======================================================== */

    [data-testid="stFileUploader"] {
        position: relative;

        z-index: 5;

        background: transparent !important;

        border: none !important;

        padding: 0 !important;
    }


    [data-testid="stFileUploaderDropzone"] {
        background:
            rgba(255,255,255,0.025) !important;

        border:
            1px solid rgba(255,255,255,0.08) !important;

        border-radius: 14px !important;

        min-height: 80px !important;
    }


    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #9ca9a2 !important;
    }


    /* ========================================================
       RESULT CARD
    ======================================================== */

    .result-card {
        padding: 25px;

        min-height: 385px;
    }


    .result-header {
        display: flex;

        align-items: center;

        gap: 12px;

        margin-bottom: 30px;
    }


    .result-icon {
        width: 46px;
        height: 46px;

        display: flex;

        align-items: center;

        justify-content: center;

        border-radius: 14px;

        background:
            rgba(34,197,94,0.12);

        font-size: 22px;
    }


    .result-title {
        font-size: 19px;

        font-weight: 800;
    }


    .result-subtitle {
        color: #71847a;

        font-size: 12px;

        margin-top: 3px;
    }


    .empty-result {
        text-align: center;

        padding: 45px 15px;

        color: #65776e;
    }


    .empty-icon {
        width: 85px;
        height: 85px;

        margin:
            0 auto 20px;

        display: flex;

        align-items: center;

        justify-content: center;

        border-radius: 50%;

        border:
            1px dashed rgba(255,255,255,0.17);

        font-size: 34px;
    }


    .empty-title {
        color: #b0bbb5;

        font-size: 16px;

        font-weight: 700;
    }


    .empty-text {
        margin-top: 7px;

        font-size: 12px;
    }


    /* ========================================================
       PREDICTION
    ======================================================== */

    .prediction-label {
        color: #71847a;

        font-size: 11px;

        text-transform: uppercase;

        letter-spacing: 1.2px;
    }


    .prediction-name {
        color: #f0fdf4;

        font-size: 32px;

        font-weight: 850;

        margin-top: 7px;

        margin-bottom: 22px;
    }


    .confidence-number {
        color: #4ade80;

        font-size: 42px;

        font-weight: 850;

        margin-top: 7px;
    }


    /* ========================================================
       SECTION TITLES
    ======================================================== */

    .section-title {
        position: relative;

        z-index: 2;

        font-size: 19px;

        font-weight: 800;

        margin-top: 36px;

        margin-bottom: 15px;
    }


    /* ========================================================
       METRICS
    ======================================================== */

    .metrics-card {
        position: relative;

        z-index: 2;

        display: flex;

        justify-content: space-around;

        align-items: center;

        padding: 22px;

        border-radius: 20px;

        background:
            rgba(10,28,19,0.75);

        border:
            1px solid rgba(255,255,255,0.07);

        backdrop-filter: blur(15px);
    }


    .metric {
        flex: 1;

        text-align: center;

        border-right:
            1px solid rgba(255,255,255,0.07);
    }


    .metric:last-child {
        border-right: none;
    }


    .metric-value {
        color: #4ade80;

        font-size: 26px;

        font-weight: 850;
    }


    .metric-label {
        color: #71847a;

        font-size: 11px;

        margin-top: 5px;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        position: relative;

        z-index: 2;

        display: flex;

        justify-content: space-between;

        margin-top: 55px;

        padding-top: 25px;

        border-top:
            1px solid rgba(255,255,255,0.06);

        color: #64766d;

        font-size: 12px;

        line-height: 1.6;
    }


    /* ========================================================
       MOBILE
    ======================================================== */

    @media (max-width: 800px) {

        .hero-title {
            font-size: 42px;
        }

        .nav-links {
            display: none;
        }

        .metrics-card {
            flex-direction: column;

            gap: 20px;
        }

        .metric {
            width: 100%;

            border-right: none;

            border-bottom:
                1px solid rgba(255,255,255,0.07);

            padding-bottom: 15px;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .footer {
            flex-direction: column;

            gap: 15px;
        }
    }


    /* ========================================================
       ABOUT SECTION STYLES
    ======================================================== */

    .about-section {
        position: relative;
        z-index: 2;
        margin-top: 45px;
    }

    .about-title {
        font-size: 28px;
        font-weight: 850;
        color: #f0fdf4;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }

    .about-intro-card {
        padding: 30px;
        margin-bottom: 25px;
    }

    .about-text {
        color: #b0bbb5;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 14px;
    }

    .about-subtitle {
        font-size: 18px;
        font-weight: 800;
        color: #4ade80;
        margin-top: 28px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Workflow Cards */
    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }

    .workflow-card {
        background: rgba(10, 28, 19, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        position: relative;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .workflow-card:hover {
        transform: translateY(-4px);
        border-color: rgba(74, 222, 128, 0.3);
    }

    .step-number {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #4ade80;
        font-weight: 800;
        font-size: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }

    .step-title {
        font-size: 15px;
        font-weight: 700;
        color: #f0fdf4;
        margin-bottom: 6px;
    }

    .step-desc {
        font-size: 13px;
        color: #81948a;
        line-height: 1.5;
    }

    /* Classes Cards */
    .classes-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 18px;
        margin-bottom: 25px;
    }

    .class-info-card {
        background: rgba(10, 28, 19, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .class-info-card:hover {
        transform: translateY(-3px);
        border-color: rgba(74, 222, 128, 0.3);
    }

    .class-card-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .badge-healthy {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .badge-early {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-late {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Tech Stack Badges */
    .tech-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
        margin-bottom: 25px;
    }

    .tech-badge {
        padding: 8px 16px;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.18);
        color: #86efac;
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Model Specs Grid */
    .specs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 25px;
    }

    .spec-item {
        background: rgba(8, 24, 16, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px;
    }

    .spec-label {
        font-size: 11px;
        color: #71847a;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }

    .spec-value {
        font-size: 15px;
        font-weight: 750;
        color: #f0fdf4;
    }

    /* Dataset Split Layout */
    .dataset-split-container {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 12px;
        margin-bottom: 20px;
    }

    .split-chip {
        flex: 1;
        min-width: 130px;
        background: rgba(34, 197, 94, 0.06);
        border: 1px solid rgba(34, 197, 94, 0.15);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }

    .split-num {
        font-size: 20px;
        font-weight: 800;
        color: #4ade80;
    }

    .split-lbl {
        font-size: 12px;
        color: #81948a;
        margin-top: 2px;
    }

    /* Limitation Box */
    .limitation-box {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.22);
        border-radius: 16px;
        padding: 20px;
        margin-top: 25px;
        margin-bottom: 25px;
        display: flex;
        gap: 15px;
        align-items: flex-start;
    }

    .limitation-icon {
        font-size: 24px;
        color: #fbbf24;
    }

    .limitation-text {
        font-size: 13px;
        color: #d1d5db;
        line-height: 1.6;
    }

    .about-bottom-tagline {
        text-align: center;
        color: #81948a;
        font-size: 13px;
        font-style: italic;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FALLING LEAVES
# ============================================================

st.html(
    """
    <div class="falling-leaves">

        <div class="falling-leaf leaf-1">🍃</div>
        <div class="falling-leaf leaf-2">🌿</div>
        <div class="falling-leaf leaf-3">🍃</div>
        <div class="falling-leaf leaf-4">🌿</div>
        <div class="falling-leaf leaf-5">🍃</div>
        <div class="falling-leaf leaf-6">🌿</div>
        <div class="falling-leaf leaf-7">🍃</div>
        <div class="falling-leaf leaf-8">🌿</div>

    </div>
    """
)


# ============================================================
# NAVBAR
# ============================================================

st.html(
    """
    <div class="navbar">

        <div class="brand">

            <div class="brand-icon">
                🌿
            </div>

            <div>

                <div class="brand-name">
                    Plant Disease AI
                </div>

                <div class="brand-subtitle">
                    Healthy Plants • Better Tomorrow
                </div>

            </div>

        </div>


        <div class="nav-links">

            <a href="#" class="nav-item nav-active">
                ⌂ Home
            </a>

            <a href="#about-section" class="nav-item">
                ⓘ About
            </a>

            <a href="#about-section" class="nav-item">
                ◉ Dataset
            </a>

            <a href="#about-section" class="nav-item">
                ◈ Model
            </a>

            <a href="#about-section" class="nav-item">
                ✉ Contact
            </a>

        </div>

    </div>
    """
)


# ============================================================
# MAIN COLUMNS
# ============================================================

left_col, middle_col, right_col = st.columns(
    [1.0, 1.15, 1.05],
    gap="large"
)


# ============================================================
# LEFT SIDE - HERO
# ============================================================

with left_col:

    st.html(
        """
        <div class="hero">

            <div class="hero-badge">
                ✦ AI POWERED
            </div>


            <div class="hero-title">

                Plant Disease

                <span class="hero-highlight">
                    Prediction
                </span>

            </div>


            <div class="hero-description">

                Upload a tomato leaf image and let our
                trained Convolutional Neural Network
                identify its health condition instantly.

            </div>

        </div>
        """
    )


    st.html(
        """
        <div class="feature">

            <div class="feature-icon">
                ⚡
            </div>

            <div>

                <div class="feature-title">
                    Fast & Accurate
                </div>

                <div class="feature-text">
                    Get instant AI-powered predictions
                </div>

            </div>

        </div>


        <div class="feature">

            <div class="feature-icon">
                ◉
            </div>

            <div>

                <div class="feature-title">
                    3 Disease Classes
                </div>

                <div class="feature-text">
                    Healthy • Early Blight • Late Blight
                </div>

            </div>

        </div>


        <div class="feature">

            <div class="feature-icon">
                ▥
            </div>

            <div>

                <div class="feature-title">
                    Trained on 3,600 Images
                </div>

                <div class="feature-text">
                    88.93% test accuracy
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# MIDDLE - UPLOAD
# ============================================================

with middle_col:

    st.html(
        """
        <div class="glass-card upload-card">

            <div class="upload-title">
                📤 Upload Tomato Leaf
            </div>

            <div class="upload-description">
                Upload a clear image of a tomato leaf
                for AI-based disease classification.
            </div>


            <div class="upload-zone">

                <div class="upload-icon">
                    ☁️
                </div>

                <div class="upload-main">
                    Choose a Leaf Image
                </div>

                <div class="upload-small">
                    JPG • JPEG • PNG
                </div>

            </div>

        </div>
        """
    )


    uploaded_file = st.file_uploader(
        "Upload tomato leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )


# ============================================================
# RIGHT - PREDICTION RESULT
# ============================================================

with right_col:

    if uploaded_file is None:

        st.html(
            """
            <div class="glass-card result-card">

                <div class="result-header">

                    <div class="result-icon">
                        🌿
                    </div>

                    <div>

                        <div class="result-title">
                            Prediction Result
                        </div>

                        <div class="result-subtitle">
                            Upload an image to see the prediction
                        </div>

                    </div>

                </div>


                <div class="empty-result">

                    <div class="empty-icon">
                        🍃
                    </div>

                    <div class="empty-title">
                        No image uploaded yet
                    </div>

                    <div class="empty-text">
                        Your prediction result will appear here
                    </div>

                </div>

            </div>
            """
        )

    else:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")


        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        resized_image = image.resize((128, 128))

        image_array = np.array(
            resized_image
        ).astype("float32")

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            float(
                predictions[predicted_index]
            ) * 100
        )


        # ----------------------------------------------------
        # DISPLAY NAME
        # ----------------------------------------------------

        display_name = (
            predicted_class
            .replace("_", " ")
            .title()
        )


        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        st.html(
            f"""
            <div class="glass-card result-card">

                <div class="result-header">

                    <div class="result-icon">
                        🌿
                    </div>

                    <div>

                        <div class="result-title">
                            Prediction Result
                        </div>

                        <div class="result-subtitle">
                            CNN classification output
                        </div>

                    </div>

                </div>


                <div class="prediction-label">
                    DETECTED CONDITION
                </div>


                <div class="prediction-name">
                    {display_name}
                </div>


                <div class="prediction-label">
                    CONFIDENCE
                </div>


                <div class="confidence-number">
                    {confidence:.2f}%
                </div>

            </div>
            """
        )


        st.progress(
            min(confidence / 100, 1.0),
            text=f"AI Confidence: {confidence:.2f}%"
        )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.html(
        """
        <div class="section-title">
            📷 Image Analysis
        </div>
        """
    )


    image_col, probability_col = st.columns(
        [1, 1.25],
        gap="large"
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_col:

        st.image(
            image,
            caption="Uploaded tomato leaf",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    with probability_col:

        st.html(
            """
            <div class="glass-card"
                 style="padding:25px;">

                <div style="
                    font-size:19px;
                    font-weight:800;
                    margin-bottom:18px;
                ">
                    📊 Prediction Breakdown
                </div>

            </div>
            """
        )


        for name, probability in zip(
            class_names,
            predictions
        ):

            display_name = (
                name
                .replace("_", " ")
                .title()
            )

            percentage = (
                float(probability) * 100
            )


            st.write(
                f"**{display_name}**"
            )


            st.progress(
                float(probability),
                text=f"{percentage:.2f}%"
            )


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.html(
    """
    <div class="section-title">
        📊 Model Overview
    </div>


    <div class="metrics-card">

        <div class="metric">

            <div class="metric-value">
                3,600
            </div>

            <div class="metric-label">
                Training Images
            </div>

        </div>


        <div class="metric">

            <div class="metric-value">
                3
            </div>

            <div class="metric-label">
                Disease Classes
            </div>

        </div>


        <div class="metric">

            <div class="metric-value">
                88.93%
            </div>

            <div class="metric-label">
                Test Accuracy
            </div>

        </div>


        <div class="metric">

            <div class="metric-value">
                CNN
            </div>

            <div class="metric-label">
                Deep Learning Model
            </div>

        </div>

    </div>
    """
)


# ============================================================
# RESULT MESSAGE
# ============================================================

if uploaded_file is not None:

    if predicted_class == "healthy":

        st.success(
            f"🌱 The model classified this image as "
            f"**Healthy** with {confidence:.2f}% confidence."
        )

    elif predicted_class == "early_blight":

        st.warning(
            f"🍂 The model classified this image as "
            f"**Early Blight** with {confidence:.2f}% confidence."
        )

    elif predicted_class == "late_blight":

        st.error(
            f"🍂 The model classified this image as "
            f"**Late Blight** with {confidence:.2f}% confidence."
        )


    st.caption(
        "AI-based classification for demonstration purposes. "
        "This system is not a substitute for professional agricultural diagnosis."
    )


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.html(
    """
    <div class="about-section" id="about-section">

        <div class="glass-card about-intro-card">

            <div class="about-title">
                About Plant Disease Prediction
            </div>

            <div class="about-text">
                This project is an AI-powered image classification system designed to identify tomato leaf conditions from an uploaded image. It uses a Convolutional Neural Network (CNN) trained on tomato leaf images to classify the leaf into three categories: Healthy, Early Blight, or Late Blight.
            </div>

            <div class="about-subtitle">
                🎯 Project Objective
            </div>

            <div class="about-text">
                The main objective is to demonstrate how Machine Learning and Deep Learning can be applied to image classification for agricultural applications. The system provides a predicted class along with its confidence score to help users understand the model's prediction.
            </div>

            <!-- HOW IT WORKS -->
            <div class="about-subtitle">
                🔄 How It Works
            </div>

            <div class="workflow-grid">

                <div class="workflow-card">
                    <div class="step-number">1</div>
                    <div class="step-title">Upload Image</div>
                    <div class="step-desc">Upload a clear image of a tomato leaf.</div>
                </div>

                <div class="workflow-card">
                    <div class="step-number">2</div>
                    <div class="step-title">Image Processing</div>
                    <div class="step-desc">The image is resized to 128 × 128 pixels and converted into RGB format.</div>
                </div>

                <div class="workflow-card">
                    <div class="step-number">3</div>
                    <div class="step-title">CNN Prediction</div>
                    <div class="step-desc">The trained CNN model analyzes visual patterns in the leaf image.</div>
                </div>

                <div class="workflow-card">
                    <div class="step-number">4</div>
                    <div class="step-title">Result</div>
                    <div class="step-desc">The system displays the predicted class and confidence score.</div>
                </div>

            </div>

            <!-- TARGET CLASSES -->
            <div class="about-subtitle">
                🏷️ Disease Classes
            </div>

            <div class="classes-grid">

                <div class="class-info-card">
                    <span class="class-card-badge badge-healthy">Healthy</span>
                    <div class="step-title">Healthy</div>
                    <div class="step-desc">Tomato leaf appears healthy.</div>
                </div>

                <div class="class-info-card">
                    <span class="class-card-badge badge-early">Early Blight</span>
                    <div class="step-title">Early Blight</div>
                    <div class="step-desc">Identifies visual patterns associated with early blight.</div>
                </div>

                <div class="class-info-card">
                    <span class="class-card-badge badge-late">Late Blight</span>
                    <div class="step-title">Late Blight</div>
                    <div class="step-desc">Identifies visual patterns associated with late blight.</div>
                </div>

            </div>

            <!-- MODEL INFO -->
            <div class="about-subtitle">
                🤖 Model Information
            </div>

            <div class="specs-grid">

                <div class="spec-item">
                    <div class="spec-label">Model</div>
                    <div class="spec-value">Convolutional Neural Network (CNN)</div>
                </div>

                <div class="spec-item">
                    <div class="spec-label">Framework</div>
                    <div class="spec-value">TensorFlow / Keras</div>
                </div>

                <div class="spec-item">
                    <div class="spec-label">Input Size</div>
                    <div class="spec-value">128 × 128 RGB</div>
                </div>

                <div class="spec-item">
                    <div class="spec-label">Number of Classes</div>
                    <div class="spec-value">3</div>
                </div>

                <div class="spec-item">
                    <div class="spec-label">Dataset Size</div>
                    <div class="spec-value">3,600 images</div>
                </div>

                <div class="spec-item">
                    <div class="spec-label">Test Accuracy</div>
                    <div class="spec-value">88.93%</div>
                </div>

            </div>

            <!-- TECHNOLOGY STACK -->
            <div class="about-subtitle">
                🛠️ Technology Stack
            </div>

            <div class="tech-badges">
                <div class="tech-badge">🐍 Python</div>
                <div class="tech-badge">⚡ TensorFlow</div>
                <div class="tech-badge">🧠 Keras</div>
                <div class="tech-badge">🔢 NumPy</div>
                <div class="tech-badge">🖼️ Pillow</div>
                <div class="tech-badge">🔬 Scikit-learn</div>
                <div class="tech-badge">🌐 Streamlit</div>
                <div class="tech-badge">📊 Matplotlib</div>
            </div>

            <!-- DATASET & SPLIT -->
            <div class="about-subtitle">
                📂 Dataset
            </div>

            <div class="about-text">
                To build the classifier, a tomato leaf image dataset was prepared using three selected classes: Healthy, Early Blight, and Late Blight. The dataset contains 3,600 images divided into training, validation, and testing sets.
            </div>

            <div class="dataset-split-container">

                <div class="split-chip">
                    <div class="split-num">2,519</div>
                    <div class="split-lbl">Training Images</div>
                </div>

                <div class="split-chip">
                    <div class="split-num">539</div>
                    <div class="split-lbl">Validation Images</div>
                </div>

                <div class="split-chip">
                    <div class="split-num">542</div>
                    <div class="split-lbl">Testing Images</div>
                </div>

            </div>

            <!-- MODEL PERFORMANCE -->
            <div class="about-subtitle">
                📈 Model Performance
            </div>

            <div class="about-text">
                <strong>Test Accuracy: 88.93%</strong><br>
                Best validation accuracy achieved during training was approximately 92.02%.
            </div>

            <div class="about-text" style="font-size:13px; color:#81948a;">
                Performance may vary depending on image quality, lighting, leaf orientation, background, and similarity between disease classes.
            </div>

            <!-- BOTTOM TAGLINE -->
            <div class="about-bottom-tagline">
                Built as a Machine Learning image classification project using TensorFlow, Keras and Streamlit.
            </div>

        </div>

    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <div>

            <strong>
                🌿 Plant Disease AI
            </strong>

            <br>

            TensorFlow • Keras • Streamlit

        </div>


        <div>

            AI for Agriculture
            &nbsp; | &nbsp;
            Detect
            &nbsp; | &nbsp;
            Prevent
            &nbsp; | &nbsp;
            Grow

        </div>

    </div>
    """
)