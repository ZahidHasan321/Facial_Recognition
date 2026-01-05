import os
import io
from pathlib import Path

# ------------------------------------------------------------------
# CRITICAL FIXES
# ------------------------------------------------------------------
# Fix for segmentation fault: disable OpenMP threading before importing face_recognition
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import streamlit as st
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# UI CONFIGURATION & CSS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Face Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI CSS
st.markdown("""
<style>
    /* Global Clean Look */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Card Styling for Containers */
    .css-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #363940;
    }

    /* Custom Header */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #4F8BF9;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    
    /* File Uploader styling */
    [data-testid='stFileUploader'] {
        border: 1px dashed #4F8BF9;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px;
        color: #adb5bd;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# HELPER FUNCTIONS (Logic Preserved)
# ------------------------------------------------------------------

@st.cache_resource
def load_known_faces():
    """Load all known face encodings and names from the known_faces directory."""
    known_face_encodings = []
    known_face_names = []
    known_face_images = {}

    if not os.path.exists(KNOWN_FACES_DIR):
        return [], [], {}

    for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)

        if filename.startswith(".") or not os.path.isfile(filepath):
            continue

        try:
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                continue

            encoding = encodings[0]
            name = os.path.splitext(filename)[0]

            known_face_encodings.append(encoding)
            known_face_names.append(name)
            known_face_images[name] = filepath

        except Exception as e:
            continue

    return known_face_encodings, known_face_names, known_face_images


def save_known_face(name, image_file):
    """Save a new known face to the known_faces directory."""
    ext = os.path.splitext(image_file.name)[1].lower()
    if not ext or ext not in ['.jpg', '.jpeg', '.png']:
        ext = '.jpg'

    filepath = os.path.join(KNOWN_FACES_DIR, f"{name}{ext}")
    with open(filepath, "wb") as f:
        f.write(image_file.getbuffer())
    return filepath


def delete_known_face(name):
    """Delete a known face from the known_faces directory."""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        filepath = os.path.join(KNOWN_FACES_DIR, f"{name}{ext}")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    return False


def recognize_faces(image, known_encodings, known_names):
    """Recognize faces in an image."""
    if isinstance(image, Image.Image):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_np = np.array(image)
    else:
        image_np = image

    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    if not face_encodings:
        return None, []

    pil_image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_image)
    results = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        results.append(name)

        # Draw box and label
        top, right, bottom, left = face_location
        color = "#00C851" if name != "Unknown" else "#ff4444"
        
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=4)
        draw.rectangle([(left, bottom), (right, bottom + 35)], fill=color)
        
        # Simple default font
        font = ImageFont.load_default()
        # To make text bigger with default font is hard, so we just draw it cleanly
        draw.text((left + 6, bottom + 6), name, fill="white")

    return pil_image, results

# ------------------------------------------------------------------
# APP LOGIC
# ------------------------------------------------------------------

# 1. Load Data
known_encodings, known_names, known_images = load_known_faces()

# 2. Sidebar UI
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3253/3253276.png", width=80)
    st.title("FaceID System")
    st.markdown("---")
    
    page = st.radio(
        "Navigation", 
        ["Scanner", "Database"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Stats")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Faces", len(known_names))
    col_s2.metric("Status", "Online")
    
    st.info(
        "**Tip:** Use high-quality images with good lighting for best accuracy.",
        icon="💡"
    )

# 3. Main Page Logic
if page == "Scanner":
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🕵️‍♂️ Face Scanner")
        st.markdown("Upload an image to detect and identify people.")
    
    # Upload Section
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop image here or click to upload", 
        type=['jpg', 'jpeg', 'png'],
        help="Supported: JPG, PNG"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Layout: Split Input vs Output
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Original Input")
            st.image(image, use_container_width=True, caption="Uploaded Image")

        with c2:
            st.subheader("Analysis Result")
            if not known_names:
                st.warning("⚠️ Database is empty. Please add faces in the 'Database' tab.")
            else:
                with st.spinner("Scanning biometrics..."):
                    annotated_image, results = recognize_faces(image, known_encodings, known_names)

                if annotated_image is None:
                    st.error("No human faces detected.")
                else:
                    st.image(annotated_image, use_container_width=True, caption="Processed Image")
                    
                    # Results list
                    st.markdown("### Identified:")
                    for name in results:
                        if name == "Unknown":
                            st.markdown(f"🔴 **Unknown Person**")
                        else:
                            st.markdown(f"🟢 **{name}**")

elif page == "Database":
    st.title("🗂️ Face Database")
    st.markdown(f"Manage the list of {len(known_names)} known individuals.")
    
    tab_add, tab_view = st.tabs(["➕ Add New Person", "👥 View Registry"])

    # TAB 1: ADD FACE
    with tab_add:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("#### 1. Person Details")
            new_name = st.text_input("Full Name", placeholder="e.g. Elon Musk")
            
        with c2:
            st.markdown("#### 2. Biometric Data")
            new_image = st.file_uploader("Face Photo", type=['jpg', 'png'])

        if new_image:
            st.image(new_image, width=200, caption="Preview")

        st.markdown("---")
        if st.button("Save to Database", type="primary", use_container_width=True):
            if not new_name or not new_image:
                st.toast("Please fill in name and upload photo.", icon="❌")
            elif new_name in known_names:
                st.toast(f"'{new_name}' already exists in database.", icon="⚠️")
            else:
                try:
                    # Validate face
                    temp_img = face_recognition.load_image_file(new_image)
                    encs = face_recognition.face_encodings(temp_img)
                    
                    if len(encs) == 0:
                        st.toast("No face detected! Try a clearer photo.", icon="❌")
                    else:
                        save_known_face(new_name, new_image)
                        load_known_faces.clear() # Reset cache
                        st.toast(f"Successfully added {new_name}!", icon="✅")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: VIEW FACES
    with tab_view:
        if not known_names:
            st.info("Database is empty.")
        else:
            # Grid Layout for faces
            cols = st.columns(4) # 4 columns grid
            for idx, name in enumerate(known_names):
                with cols[idx % 4]:
                    st.markdown('<div class="css-card" style="padding:10px; text-align:center;">', unsafe_allow_html=True)
                    
                    # Display Image
                    img_path = known_images.get(name)
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    
                    st.markdown(f"**{name}**")
                    
                    # Delete Button
                    if st.button("Remove", key=f"del_{name}"):
                        delete_known_face(name)
                        load_known_faces.clear()
                        st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
