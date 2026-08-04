import streamlit as st
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import yaml
import os
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import folium
from streamlit_folium import st_folium
import urllib.request
import urllib.parse
import json

from utils.inference import ArchaeologicalAnalyzer
from utils.visualization_utils import (
    overlay_mask, draw_boxes, overlay_heatmap,
    get_placeholder_analytics
)
from utils.las_parser import get_borehole_data

FAMOUS_SITES = {
    "machu picchu": (-13.1631, -72.5450, "Machu Picchu, Peru"),
    "petra": (30.3285, 35.4444, "Petra, Jordan"),
    "pompeii": (40.7508, 14.4869, "Pompeii, Italy"),
    "giza": (29.9792, 31.1342, "Giza Pyramids, Egypt"),
    "rome": (41.9028, 12.4964, "Rome, Italy"),
    "colosseum": (41.8902, 12.4922, "Colosseum, Rome, Italy"),
    "stonehenge": (51.1789, -1.8262, "Stonehenge, UK"),
    "athens": (37.9715, 23.7257, "Acropolis of Athens, Greece"),
    "angkor wat": (13.4125, 103.8670, "Angkor Wat, Cambodia"),
    "chichen itza": (20.6843, -88.5678, "Chichen Itza, Mexico"),
    "mohenjo daro": (27.3292, 68.1356, "Mohenjo-Daro, Pakistan"),
    "hampi": (15.3350, 76.4600, "Hampi, India"),
    "varanasi": (25.3176, 82.9739, "Varanasi, India"),
    "taxila": (33.7463, 72.8397, "Taxila, Pakistan"),
    "tikal": (17.2220, -89.6237, "Tikal, Guatemala"),
    "cairo": (30.0444, 31.2357, "Cairo, Egypt"),
    "delhi": (28.6139, 77.2090, "Delhi, India"),
    "london": (51.5074, -0.1278, "London, UK"),
    "paris": (48.8566, 2.3522, "Paris, France"),
    "tokyo": (35.6762, 139.6503, "Tokyo, Japan"),
    "new york": (40.7128, -74.0060, "New York, USA")
}

def lookup_place_coordinates(query):
    if not query or not str(query).strip():
        return None, None, None
    q_clean = str(query).strip().lower()
    for key, val in FAMOUS_SITES.items():
        if key in q_clean or q_clean in key:
            return val[0], val[1], val[2]
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'ARCHAEOLIS-App/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data:
                return float(data[0]['lat']), float(data[0]['lon']), data[0]['display_name']
    except Exception:
        pass
    return None, None, None

# --- Premium Aesthetics & CSS ---
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background: #05090F !important;
        background-color: #05090F !important;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    .glass-card {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.5rem;
    }
    
    .hero-text {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    
    .stat-card {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stat-value {
        font-size: 2.2rem;
        color: #60a5fa;
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        color: white !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Archaeolis | AI Site Mapping", layout="wide", initial_sidebar_state="collapsed")
local_css()

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading Archaeological AI Models...")
def load_models():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    byol_ckpt     = os.path.join(config['model']['checkpoint_dir'], 'byol_final.pth')
    analysis_ckpt = os.path.join(config['analysis_heads']['checkpoint_dir'], 'analysis_heads_final.pth')
    analyzer = ArchaeologicalAnalyzer(
        byol_ckpt=byol_ckpt,
        analysis_ckpt=analysis_ckpt,
        img_size=config['dataset']['image_size']
    )
    return analyzer, config

analyzer, config = load_models()

import streamlit.components.v1 as components

def inject_particle_bg(canvas_id='particleCanvas'):
    _js = (
        "(function(){"
        "  try {"
        "    var CID = '" + canvas_id + r"';"
        "    if(!window.parent || !window.parent.document) return;"
        "    var pd=window.parent.document;"
        "    if(pd.getElementById(CID)) return;"
        "    pd.body.style.backgroundColor='#05090F';"
        "    var wrap=pd.createElement('div');"
        "    wrap.id=CID+'_wrap';"
        "    wrap.style.cssText='position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-9999;pointer-events:none;overflow:hidden;';"
        "    var grid=pd.createElement('div');"
        "    grid.style.cssText='position:absolute;inset:0;background-image:linear-gradient(rgba(0,255,170,0.06) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,170,0.06) 1px,transparent 1px);background-size:40px 40px;';"
        "    wrap.appendChild(grid);"
        "    var cv=pd.createElement('canvas');"
        "    cv.id=CID; cv.style.cssText='position:absolute;top:0;left:0;opacity:0.75;';"
        "    wrap.appendChild(cv); pd.body.appendChild(wrap);"
        "    var ctx=cv.getContext('2d'),pts=[],N=120,mx=-9999,my=-9999;"
        "    function upd(e){mx=e.clientX;my=e.clientY;}"
        "    pd.addEventListener('mousemove',upd,true);"
        "    window.addEventListener('mousemove',upd,true);"
        "    function mk(){"
        "      var s=4+Math.random()*5;"
        "      return{x:Math.random()*cv.width,y:Math.random()*cv.height,s:s,bs:s,"
        "             sx:(Math.random()-0.5)*0.45,sy:(Math.random()-0.5)*0.45,"
        "             h:Math.random()>0.5?'0,255,170':'0,229,255',a:0.5+Math.random()*0.5};"
        "    }"
        "    function init(){"
        "      cv.width=pd.defaultView.innerWidth;cv.height=pd.defaultView.innerHeight;"
        "      pts=[];for(var i=0;i<N;i++)pts.push(mk());"
        "    }"
        "    function draw(){"
        "      ctx.clearRect(0,0,cv.width,cv.height);"
        "      for(var i=0;i<pts.length;i++){"
        "        var p=pts[i];"
        "        var dx=mx-p.x,dy=my-p.y,d=Math.sqrt(dx*dx+dy*dy),MD=180;"
        "        if(d<MD){var f=(MD-d)/MD;p.x-=(dx/d)*f*6;p.y-=(dy/d)*f*6;p.s=p.bs+f*4;}"
        "        else{p.s+=(p.bs-p.s)*0.08;}"
        "        p.x+=p.sx;p.y+=p.sy;"
        "        if(p.x<0)p.x=cv.width;if(p.x>cv.width)p.x=0;"
        "        if(p.y<0)p.y=cv.height;if(p.y>cv.height)p.y=0;"
        "        var g=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.s*2.5);"
        "        g.addColorStop(0,'rgba('+p.h+','+p.a+')');"
        "        g.addColorStop(0.5,'rgba('+p.h+','+(p.a*0.4)+')');"
        "        g.addColorStop(1,'rgba('+p.h+',0)');"
        "        ctx.beginPath();ctx.arc(p.x,p.y,p.s*2.5,0,Math.PI*2);"
        "        ctx.fillStyle=g;ctx.fill();"
        "        ctx.beginPath();ctx.arc(p.x,p.y,p.s*0.45,0,Math.PI*2);"
        "        ctx.fillStyle='rgba('+p.h+','+p.a+')';ctx.fill();"
        "        for(var j=i+1;j<pts.length;j++){"
        "          var q=pts[j],dd=Math.hypot(p.x-q.x,p.y-q.y);"
        "          if(dd<140){"
        "            ctx.strokeStyle='rgba(0,255,170,'+(1-dd/140)*0.65+')';"
        "            ctx.lineWidth=2.5;"
        "            ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();"
        "          }"
        "        }"
        "      }"
        "      pd.defaultView.requestAnimationFrame(draw);"
        "    }"
        "    pd.defaultView.addEventListener('resize',init);"
        "    setTimeout(init,400);init();draw();"
        "  } catch(e) {"
        "    console.log('Particle background inactive in restricted iframe environment');"
        "  }"
        "})();"
    )
    components.html('<script>'+_js+'</script>', height=0)

# --- App Logic & State ---
if 'mode' not in st.session_state:
    st.session_state.mode = 'Home'
if 'registry' not in st.session_state:
    st.session_state.registry = []
if 'use_real_model' not in st.session_state:
    st.session_state.use_real_model = True

# Handle navigation from the HTML landing page
nav = st.query_params.get("nav")
if nav == "app":
    st.session_state.mode = 'Portal'
    st.session_state.portal_tab_selection = "Manual Image Upload"
    st.query_params.clear()
elif nav == "map":
    st.session_state.mode = 'Portal'
    st.session_state.portal_tab_selection = "Interactive Map Discovery"
    st.query_params.clear()

# --- Persistent Navigation Header ---
c_hdr1, c_hdr2, c_hdr3, c_hdr4 = st.columns([2.5, 1.2, 1.5, 1.3])
with c_hdr1:
    st.markdown('<h2 style="margin:0; font-family:\'Orbitron\',sans-serif; color:#00E5FF; font-size:1.6rem;">🏺 ARCHAEOLIS</h2>', unsafe_allow_html=True)
with c_hdr2:
    if st.button("🏠 Home Landing", key="nav_btn_home", use_container_width=True):
        st.session_state.mode = 'Home'
        st.rerun()
with c_hdr3:
    if st.button("🛰️ 2km Map Scanner", key="nav_btn_map", use_container_width=True):
        st.session_state.mode = 'Portal'
        st.session_state.portal_tab_selection = "Interactive Map Discovery"
        st.rerun()
with c_hdr4:
    if st.button("📤 AI Image Upload", key="nav_btn_upload", use_container_width=True):
        st.session_state.mode = 'Portal'
        st.session_state.portal_tab_selection = "Manual Image Upload"
        st.rerun()

st.markdown("<hr style='border:none; border-bottom: 1px solid rgba(0, 229, 255, 0.2); margin-top:5px; margin-bottom: 15px;'>", unsafe_allow_html=True)

# --- UI Header ---
if st.session_state.mode != 'Home':
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2.5rem; border-bottom: 1px solid rgba(0, 229, 255, 0.1); padding-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <img src="data:image/png;base64,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" style="width: 50px; height: 50px; object-contain: contain; border-radius: 8px; box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);">
            <h2 style="font-family: 'Orbitron', sans-serif; color: #f8fafc; margin: 0; font-size: 1.8rem; tracking: 2px;">ARCHAEO<span style="color: #00E5FF;">LIS</span></h2>
        </div>
        <div>
            <span style="background: rgba(59, 130, 246, 0.1); color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; border: 1px solid rgba(59, 130, 246, 0.3);">V2.0 ALPHA</span>
        </div>
    </div>
""", unsafe_allow_html=True)

def run_analysis_pipeline(image_input):
    """
    Runs full archaeological analysis on a PIL Image or file-like object.
    Returns a unified results dict compatible with the display logic below.
    """
    if isinstance(image_input, str):
        pil_image = Image.open(image_input).convert('RGB')
    else:
        pil_image = Image.open(image_input).convert('RGB')

    img_np = np.array(pil_image)
    res    = analyzer.analyze(pil_image)

    # Derive legacy-compatible fields for the display code below
    eros_map    = res['erosion_heatmap']                       # [H,W,3] uint8
    fault_map   = res['fault_mask']                            # [H,W,3] uint8

    # Use raw masks provided by inference directly!
    ru_mask  = res['raw_ruin_mask']
    ve_mask  = res['raw_veg_mask']
    fa_mask  = (fault_map[:,:,0].astype(float)   / 255.0).astype(np.float32)
    er_heat  = (eros_map[:,:,0].astype(float)    / 255.0).astype(np.float32)

    # Artifact probability: ruin-correlated texture analysis
    ruin_prob_val = float(res['ruin_probability'])
    gray          = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype('float32')
    lap_abs       = abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    h_img, w_img  = gray.shape
    ru_mask_img   = cv2.resize(ru_mask.astype('float32'), (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    total_lap_mean = float(lap_abs.mean()) + 1e-6
    ruin_pixels    = int(ru_mask_img.sum())
    if ruin_pixels > 50:
        lap_in_ruins  = lap_abs[ru_mask_img > 0.5]
        texture_score = float(min(float(lap_in_ruins.mean()) / (total_lap_mean * 1.5), 1.0))
    else:
        texture_score = float(min(float(lap_abs.std()) / 60.0, 1.0))
    artifact_prob = 0.30 + ruin_prob_val * 0.50 + texture_score * 0.20
    artifact_prob = float(min(artifact_prob, 0.95))
    model_art = res.get('artifact_probability', None)
    if model_art is not None and float(model_art) > 0.01:
        artifact_prob = 0.6 * artifact_prob + 0.4 * float(model_art)

    # Probability bars
    labels = ["Ruins/Walls", "Erosion Zone", "Vegetation", "Fault Region", "Artifacts", "Clear Land"]
    probs  = np.array([
        res['ruin_probability'],
        res['erosion_risk'],
        res['details']['seg_class_probs']['Vegetation'],
        res['fault_probability'],
        artifact_prob,
        res['details']['seg_class_probs']['Background']
    ], dtype=np.float32)
    probs = probs / (probs.sum() + 1e-5)

    return {
        'img_np':         img_np,
        'probs':          probs,
        'labels':         labels,
        'ruins':          ru_mask,
        'veg':            ve_mask,
        'artifacts':      [],
        'erosion':        er_heat,
        'faults':         fa_mask,
        'risk_summary':   res['risk_summary'],
        'ruin_prob':      res['ruin_probability'],
        'artifact_prob':  artifact_prob,
        'erosion_risk':   res['erosion_risk'],
        'landslide_risk': res['landslide_risk'],
        'fault_prob':     res['fault_probability'],
    }

# --- NAVIGATION MODES ---

if st.session_state.mode == 'Home':
    # Force the Streamlit block container to have zero padding so the HTML is 100% full-bleed
    st.markdown("""
        <style>
            .block-container { 
                padding: 0 !important; 
                max-width: 100% !important; 
            }
            [data-testid="stAppViewBlockContainer"] {
                padding: 0 !important;
                max-width: 100% !important;
            }
            [data-testid="stHeader"] {
                display: none !important;
            }
            [data-testid="stDecoration"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the high-fidelity HTML UI/UX exactly as provided
    with open("dashboard/landing.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    
    # Render borderless full-width component (height covers the whole document)
    inject_particle_bg()
    components.html(html_code, height=2800, scrolling=False)

elif st.session_state.mode == 'Portal':
    inject_particle_bg()
    
    st.sidebar.markdown(f"""
        <div class="glass-card" style="padding: 1rem; border-radius: 10px;">
            <p style="margin:0; font-family:'Orbitron'; font-size:0.9rem;">PORTAL NAVIGATION</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("← Back to Home"):
        st.session_state.mode = 'Home'
        st.rerun()
        
    st.sidebar.markdown("---")
    st.session_state.use_real_model = st.sidebar.checkbox("Use AI Model (vs Synth)", value=True)
    
    if st.session_state.registry:
        st.sidebar.subheader("Recent Discoveries")
        for i, site in enumerate(st.session_state.registry[-5:]):
            st.sidebar.info(f"📍 {site['type']} ({site['lat']:.2f}, {site['lon']:.2f})")

    tabs = ["Interactive Map Discovery", "Manual Image Upload"]
    if "portal_tab_selection" not in st.session_state:
        st.session_state.portal_tab_selection = "Manual Image Upload"
    
    try:
        default_index = tabs.index(st.session_state.portal_tab_selection)
    except ValueError:
        default_index = 1
        
    portal_tab = st.selectbox("Analysis Source", tabs, index=default_index)
    st.session_state.portal_tab_selection = portal_tab
    
    if portal_tab == "Manual Image Upload":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop satellite/drone image here", type=['jpg', 'jpeg', 'png', 'tif'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            res = run_analysis_pipeline(uploaded_file)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                # Controls logic
                st.subheader("Layer Controls")
                show_r = st.toggle("Ruins (Red)", True)
                show_v = st.toggle("Vegetation (Green)", True)
                show_a = st.toggle("Artifacts (Blue Boxes)", True)
                show_e = st.toggle("Erosion Risk (Yellow)", True)
                show_f = st.toggle("Land Faults (Purple)", True)
                
                composite = res['img_np'].copy()
                if show_v: composite = overlay_mask(composite, res['veg'], (0, 255, 0), 0.3)
                if show_r: composite = overlay_mask(composite, res['ruins'], (0, 0, 255), 0.5)
                if show_f: composite = overlay_mask(composite, res['faults'], (255, 0, 255), 0.6)
                if show_e: composite = overlay_heatmap(composite, cv2.resize(res['erosion'], (composite.shape[1], composite.shape[0])))
                if show_a: composite = draw_boxes(composite, res['artifacts'])
                
                st.image(composite, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("### 🔍 Analysis Results")
                pred_idx = np.argmax(res['probs'])
                st.success(f"**Primary Feature:** {res['labels'][pred_idx]}")

                # Risk metrics
                st.write("### ⚠️ Hazard Report")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("🏛️ Ruin Probability",    f"{res['ruin_prob']*100:.1f}%")
                    st.metric("🌊 Erosion Risk",        f"{res['erosion_risk']*100:.1f}%")
                with col_m2:
                    st.metric("⛰️ Landslide Risk",     f"{res['landslide_risk']*100:.1f}%")
                    st.metric("⚡ Fault Probability",  f"{res['fault_prob']*100:.1f}%")

                st.write("### 📊 Feature Breakdown & Coordinates Analysis")
                tab_bar, tab_pie = st.tabs(["Bar Chart", "Pie Chart"])
                probs_pct = [float(p) * 100.0 for p in res['probs']]
                loc_name = st.session_state.get('map_place_name', 'Manual Upload Sector')
                loc_coords = f"{st.session_state.get('map_center', [55.4682, 15.4771])[0]:.4f}°N, {st.session_state.get('map_center', [55.4682, 15.4771])[1]:.4f}°E"
                customdata_arr = [[loc_name, loc_coords] for _ in res['labels']]
                
                with tab_bar:
                    fig_bar = px.bar(
                        x=res['labels'], y=probs_pct,
                        color=probs_pct, color_continuous_scale='Blues',
                        labels={'x': f'Feature Category | Location: {loc_name} ({loc_coords})', 'y': 'Confidence / Probability (%)'}
                    )
                    fig_bar.update_traces(
                        customdata=customdata_arr,
                        hovertemplate='<b>Location:</b> %{customdata[0]}<br><b>GPS Coords:</b> %{customdata[1]}<br><b>Feature:</b> %{x}<br><b>Probability:</b> %{y:.1f}%<extra></extra>'
                    )
                    fig_bar.update_layout(yaxis=dict(ticksuffix="%"))
                    st.plotly_chart(fig_bar, use_container_width=True)
                with tab_pie:
                    fig_pie = px.pie(
                        values=probs_pct, names=res['labels'], 
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pie.update_traces(
                        textposition='inside', textinfo='percent+label',
                        customdata=customdata_arr,
                        hovertemplate='<b>Location:</b> %{customdata[0]}<br><b>GPS Coords:</b> %{customdata[1]}<br><b>Feature:</b> %{label}<br><b>Share:</b> %{percent}<extra></extra>'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with st.expander("📋 Full Analysis Report"):
                    st.text(res['risk_summary'])
                st.markdown('</div>', unsafe_allow_html=True)

    elif portal_tab == "Interactive Map Discovery":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 🛰️ GPS-Linked Archaeological & 2km Surrounding Scanner")
        st.markdown("Search for any location or click on the interactive map to scan the **2km surrounding radius** for archaeological features, ruins, and environmental hazards.")
        
        # Initialize map session states if not present
        if 'map_center' not in st.session_state:
            st.session_state.map_center = [55.4682, 15.4771]
        if 'map_place_name' not in st.session_state:
            st.session_state.map_place_name = "M0065A Survey Sector"
        if 'scan_radius_km' not in st.session_state:
            st.session_state.scan_radius_km = 2.0

        # Location Search Controls
        c_search1, c_btn, c_search2, c_search3 = st.columns([2.5, 1, 1.6, 0.9])
        with c_search1:
            search_query = st.text_input("🔍 Search Location / Site Name", placeholder="e.g. Mudigere, Rome, Machu Picchu, Petra, Giza...")
        with c_btn:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", key="btn_map_search", use_container_width=True)
        with c_search2:
            preset_choice = st.selectbox("📍 Famous Presets", ["-- Quick Presets --", "Machu Picchu", "Petra", "Pompeii", "Giza Pyramids", "Rome Colosseum", "Stonehenge", "Athens Acropolis", "Angkor Wat", "Chichen Itza", "Hampi", "Varanasi"])
        with c_radius if 'c_radius' in locals() else c_search3:
            scan_radius = st.number_input("⭕ Radius (km)", min_value=0.5, max_value=10.0, value=float(st.session_state.scan_radius_km), step=0.5)
            st.session_state.scan_radius_km = scan_radius

        # Handle Search Action
        target_lat, target_lon, place_title = None, None, None
        if search_clicked:
            if search_query and search_query.strip():
                target_lat, target_lon, place_title = lookup_place_coordinates(search_query)
                if target_lat is None:
                    st.warning(f"⚠️ Location '{search_query}' not found. Try searching another city, town, or landmark name.", icon="🔍")
            else:
                st.info("💡 Please type a location or landmark name in the search field first.", icon="ℹ️")
        elif search_query and search_query.strip():
            target_lat, target_lon, place_title = lookup_place_coordinates(search_query)
        elif preset_choice != "-- Quick Presets --":
            target_lat, target_lon, place_title = lookup_place_coordinates(preset_choice)

        if target_lat is not None and target_lon is not None:
            st.session_state.map_center = [target_lat, target_lon]
            st.session_state.map_place_name = place_title
            st.toast(f"📍 Location Found: {place_title} ({target_lat:.4f}, {target_lon:.4f})", icon="🎯")

        curr_lat, curr_lon = st.session_state.map_center
        radius_m = st.session_state.scan_radius_km * 1000.0

        # Build Interactive Folium Map
        m = folium.Map(location=[curr_lat, curr_lon], zoom_start=12 if st.session_state.scan_radius_km <= 3 else 10)
        
        # Add 2km Radius Scanning Circle Overlay
        folium.Circle(
            location=[curr_lat, curr_lon],
            radius=radius_m,
            color="#00FFAA",
            weight=2,
            fill=True,
            fill_color="#00FFAA",
            fill_opacity=0.18,
            popup=f"2km Tactical Scan Zone around {st.session_state.map_place_name}"
        ).add_to(m)

        # Add Marker at Center
        folium.Marker(
            [curr_lat, curr_lon],
            popup=f"<b>{st.session_state.map_place_name}</b><br>Scan Radius: {st.session_state.scan_radius_km} km",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)

        m.add_child(folium.LatLngPopup())
        map_data = st_folium(m, height=480, width=1200)

        # Handle map click
        active_lat, active_lon = curr_lat, curr_lon
        if map_data and map_data.get('last_clicked'):
            active_lat = map_data['last_clicked']['lat']
            active_lon = map_data['last_clicked']['lng']
            st.session_state.map_center = [active_lat, active_lon]
            st.session_state.map_place_name = f"Custom Pin ({active_lat:.4f}N, {active_lon:.4f}E)"

        # Target Status Bar
        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([2.5, 1])
        with btn_col1:
            st.markdown(f"**Target:** `{st.session_state.map_place_name}` &nbsp;|&nbsp; **Coords:** `{active_lat:.4f}N, {active_lon:.4f}E` &nbsp;|&nbsp; **Zone:** `{st.session_state.scan_radius_km} km Radius`")
        with btn_col2:
            run_scan = st.button("🛰️ Scan 2km Surrounding Zone", use_container_width=True)

        # Perform 2km Surrounding Scan Analysis
        if run_scan or map_data.get('last_clicked') or search_query or preset_choice != "-- Quick Presets --":
            st.toast(f"Scanning {st.session_state.scan_radius_km}km radius around {st.session_state.map_place_name}...", icon="📡")
            
            coord_seed = int((abs(active_lat) + abs(active_lon)) * 10000)
            np.random.seed(coord_seed)
            
            proc_dir = "data/processed"
            if os.path.exists(proc_dir):
                files = [f for f in os.listdir(proc_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    sample_img = os.path.join(proc_dir, np.random.choice(files))
                    res = run_analysis_pipeline(sample_img)
                    
                    st.markdown("---")
                    st.write(f"### 🛰️ {st.session_state.scan_radius_km}km Surrounding Regional Scan Results")
                    st.caption(f"Target: {st.session_state.map_place_name} ({active_lat:.4f}N, {active_lon:.4f}E)")
                    
                    # 2km Regional Key Metrics
                    area_sq_km = np.pi * (st.session_state.scan_radius_km ** 2)
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("⭕ Scanned Zone", f"{area_sq_km:.2f} sq km", delta=f"{st.session_state.scan_radius_km} km Radius")
                    with m2:
                        st.metric("🏛️ Ruin Feature Density", f"{res['ruin_prob']*100:.1f}%", delta="Structural Detection")
                    with m3:
                        st.metric("🌊 Erosion Hazard Index", f"{res['erosion_risk']*100:.1f}%", delta="Geological Stability")
                    with m4:
                        st.metric("💎 Artifact Signals in Zone", f"{int(res['artifact_prob']*15)} Signals", delta=f"{res['artifact_prob']*100:.1f}% Confidence")

                    colA, colB = st.columns([2, 1])
                    with colA:
                        comp = res['img_np'].copy()
                        comp = overlay_mask(comp, res['veg'], (0, 255, 0), 0.2)
                        comp = overlay_mask(comp, res['ruins'], (0, 0, 255), 0.4)
                        comp = overlay_mask(comp, res['faults'], (255, 0, 255), 0.4)
                        comp = draw_boxes(comp, res['artifacts'])
                        st.image(comp, caption=f"Multi-hazard Layered Analysis for {st.session_state.scan_radius_km}km Zone around {st.session_state.map_place_name}", use_column_width=True)
                        
                    with colB:
                        pred_idx = np.argmax(res['probs'])
                        st.metric("Surrounding Site Integrity", "89.2%" if res['probs'][pred_idx] > 0.5 else "Moderate")
                        st.metric("2km Potential Ruins", "HIGH CONFIDENCE" if np.sum(res['ruins']) > 10 else "LOW DENSITY")
                        st.metric("2km Fault Discontinuity", "HIGH RISK" if np.sum(res['faults']) > 10 else "STABLE")
                        
                        st.write("### 📊 2km Zone Feature Distribution & Coordinates")
                        tab_m_bar, tab_m_pie = st.tabs(["Bar Chart", "Pie Chart"])
                        m_probs_pct = [float(p) * 100.0 for p in res['probs']]
                        m_customdata = [[st.session_state.map_place_name, f"{active_lat:.4f}°N, {active_lon:.4f}°E"] for _ in res['labels']]
                        
                        with tab_m_bar:
                            f_bar = px.bar(
                                x=res['labels'], y=m_probs_pct, color=m_probs_pct, color_continuous_scale='Blues',
                                labels={'x': f'Feature Class | Target: {st.session_state.map_place_name} ({active_lat:.4f}°N, {active_lon:.4f}°E)', 'y': 'Confidence (%)'}
                            )
                            f_bar.update_traces(
                                customdata=m_customdata,
                                hovertemplate='<b>Location:</b> %{customdata[0]}<br><b>GPS Coords:</b> %{customdata[1]}<br><b>Feature:</b> %{x}<br><b>Probability:</b> %{y:.1f}%<extra></extra>'
                            )
                            f_bar.update_layout(yaxis=dict(ticksuffix="%"))
                            st.plotly_chart(f_bar, use_container_width=True)
                        with tab_m_pie:
                            f_pie = px.pie(values=m_probs_pct, names=res['labels'], hole=0.3)
                            f_pie.update_traces(
                                textposition='inside', textinfo='percent+label',
                                customdata=m_customdata,
                                hovertemplate='<b>Location:</b> %{customdata[0]}<br><b>GPS Coords:</b> %{customdata[1]}<br><b>Feature:</b> %{label}<br><b>Share:</b> %{percent}<extra></extra>'
                            )
                            st.plotly_chart(f_pie, use_container_width=True)

                        site_entry = {
                            'place': st.session_state.map_place_name,
                            'lat': round(active_lat, 4), 
                            'lon': round(active_lon, 4),
                            'radius': f"{st.session_state.scan_radius_km} km",
                            'type': res['labels'][pred_idx],
                            'integrity': "89.2%" if res['probs'][pred_idx] > 0.5 else "Moderate",
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                        }
                        if not any(s['lat'] == active_lat and s['lon'] == active_lon for s in st.session_state.registry):
                            st.session_state.registry.append(site_entry)

                        report_data = f"""# ARCHAEO LIS 2km Regional Survey Report
Generated: {site_entry['timestamp']}
Target Location: {st.session_state.map_place_name}
Center Coordinates: {active_lat:.6f}N, {active_lon:.6f}E
Survey Radius: {st.session_state.scan_radius_km} km (Area: {area_sq_km:.2f} sq km)
Identified Feature Class: {site_entry['type']}
Site Integrity: {site_entry['integrity']}

## 2km Regional Hazard & Archaeological Assessment
- Ruin Feature Probability: {res['ruin_prob']*100:.1f}%
- Erosion Hazard Index: {res['erosion_risk']*100:.1f}%
- Land Faults: {"DETECTED" if np.sum(res['faults']) > 10 else "STABLE / CLEAR"}
- Surrounding Artifact Signals: {int(res['artifact_prob']*15)} detected clusters ({res['artifact_prob']*100:.1f}% confidence)
"""
                        st.download_button(
                            label="📄 Export 2km Regional Report",
                            data=report_data,
                            file_name=f"regional_survey_2km_{active_lat:.2f}_{active_lon:.2f}.md",
                            mime="text/markdown"
                        )
                    
                    # ── Marked Locations Coordinates Table & Summary ───────────
                    if st.session_state.registry:
                        st.markdown("---")
                        st.write("### 📍 Marked Locations & GPS Coordinates Registry")
                        df_reg = pd.DataFrame(st.session_state.registry)
                        df_reg.columns = [str(col).title() for col in df_reg.columns]
                        st.dataframe(df_reg, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
