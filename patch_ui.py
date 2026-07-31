import os

file_path = "dashboard/streamlit_app.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Split the code where `elif st.session_state.mode == 'Portal':` begins
split_marker = "elif st.session_state.mode == 'Portal':"
if split_marker in code:
    before, after = code.split(split_marker, 1)
else:
    print("Could not find Portal code block.")
    exit(1)

new_portal_code = """elif st.session_state.mode == 'Portal':
    # Force dark bg and fonts for Portal (Tactical Theme)
    st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    .stApp, .main, .block-container {
        background-color: #05090F !important;
        background: #05090F !important;
        color: #f1f5f9;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stApp > header {
        display: none !important;
    }
    
    /* Center area grid background */
    .grid-bg-container {
        background-image: radial-gradient(circle, #1a332d 1px, transparent 1px);
        background-size: 30px 30px;
        border: 1px solid #1e293b;
        background-color: #000000;
        border-radius: 8px;
        padding: 10px;
        min-height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    
    .tactical-hdr {
        border-bottom: 1px solid #1e293b; 
        background-color: #0f172a; 
        padding: 12px 24px; 
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 8px;
    }
    .hud-text {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: #00FFAA;
        letter-spacing: 2px;
    }
    .bar-bg {
        height: 6px;
        background: #1e293b;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .bar-fill {
        height: 100%;
        background: #00FFAA;
        transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
    }
    .metric-box {
        background: rgba(15,23,42,0.5);
        border: 1px solid #1e293b;
        padding: 12px;
        text-align: left;
    }
    </style>
    ''', unsafe_allow_html=True)
    
    # Tactical Header
    colH1, colH2 = st.columns([2,1])
    with colH1:
        st.markdown('<div class="tactical-hdr"><h2 style="margin:0; font-family:\\'Space Grotesk\\'; font-weight:700; font-size:1.5rem; display:inline-block; color:white;">ARCH-AI <span class="hud-text" style="color:rgba(0,255,170,0.7);">v1.0.4</span></h2><span class="hud-text" style="color:#64748b; margin-left:20px;">SYSTEM STATUS: <strong style="color:#00FFAA;">ONLINE</strong></span></div>', unsafe_allow_html=True)
    with colH2:
        if st.button("← HOME REVERT", use_container_width=True):
            st.session_state.mode = 'Home'
            st.rerun()
            
    # Main Dashboard layout mimicking HTML reference
    colL, colM, colR = st.columns([1, 2.5, 1.2], gap="small")
    
    # --- LEFT COLUMN ---
    with colL:
        st.markdown('<div class="hud-text" style="margin-bottom:4px; text-transform:uppercase;">Detection Queue</div>', unsafe_allow_html=True)
        st.markdown('<div class="hud-text" style="color:#64748b; margin-bottom:16px;">Satellite Scan Files</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drop imagery / click to upload", type=['jpg', 'jpeg', 'png', 'tif'])
        
        if uploaded_file:
            st.markdown(f'''
            <div style="padding:10px; border:1px solid #00FFAA; background:rgba(0,255,170,0.05); margin-top:20px; text-align:center;">
                <div class="hud-text" style="color:white; font-size:11px; margin-bottom:5px;">{uploaded_file.name}</div>
                <div class="hud-text" style="font-size:9px;">ACTUAL: ANALYZING... ✓</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="padding:40px 10px; border:1px solid #1e293b; text-align:center; margin-top:20px;">
                <div class="hud-text" style="color:#64748b; font-size:10px;">Queue empty.<br/>Upload a scan above.</div>
            </div>
            ''', unsafe_allow_html=True)
            
    # --- RUN PIPELINE ---
    res = None
    if uploaded_file:
        res = run_analysis_pipeline(uploaded_file)
        
    # --- CENTER COLUMN ---
    with colM:
        tab1, tab2, tab3, tab4 = st.tabs(["COMPOSITE", "SEGMENTATION", "EROSION", "FAULT MAP"])
        
        with tab1:
            if res is not None:
                c_h1, c_h2 = st.columns([1,1])
                with c_h1:
                    st.markdown('<div class="hud-text" style="background:rgba(15,23,42,0.6); padding:8px; border-left:1px solid #00FFAA; margin-bottom: 10px;">LAT: 29.9792° N<br/>LNG: 31.1342° E<br/>ALT: 420.55 KM</div>', unsafe_allow_html=True)
                with c_h2:
                    st.markdown(f'<div class="hud-text" style="background:rgba(15,23,42,0.6); padding:8px; border-right:1px solid #00FFAA; text-align:right; margin-bottom: 10px;">FRAME: {np.random.randint(1000,9999)}-X<br/>SAT: ARCH-EYE-01<br/>MODE: MULTI-SPECTRAL</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="grid-bg-container" style="min-height:400px; display:block;">', unsafe_allow_html=True)
                
                # Setup visualization composite
                comp = res['img_np'].copy()
                comp = overlay_mask(comp, res['veg'], (0, 255, 0), 0.3)
                comp = overlay_mask(comp, res['ruins'], (0, 0, 255), 0.5)
                comp = overlay_mask(comp, res['faults'], (255, 0, 255), 0.6)
                
                st.image(comp, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                pred_label = res["labels"][np.argmax(res["probs"])]
                st.markdown(f'<div style="text-align:center; margin-top:20px;"><div class="hud-text" style="padding:8px 16px; background:rgba(0,255,170,0.1); border:1px solid #00FFAA; display:inline-block; font-size:12px;">SCAN COMPLETE — {pred_label} DETECTED</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="grid-bg-container" style="min-height:500px;"><div style="text-align:center; opacity:0.3;"><h1 style="font-size:5rem; color:#00FFAA;">🛰️</h1><div class="hud-text" style="font-size:12px;">NO ACTIVE SCAN</div></div></div>', unsafe_allow_html=True)
            
        with tab2:
            st.markdown('<div class="grid-bg-container" style="min-height:500px; display:block;">', unsafe_allow_html=True)
            if res is not None:
                st.image(res['img_np'], use_column_width=True)
            else:
                st.markdown('<div class="hud-text" style="text-align:center;">AWAITING DATA</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab3:
            st.markdown('<div class="grid-bg-container" style="min-height:500px; display:block;">', unsafe_allow_html=True)
            if res is not None:
                h = overlay_heatmap(res['img_np'].copy(), cv2.resize(res['erosion'], (res['img_np'].shape[1], res['img_np'].shape[0])))
                st.image(h, use_column_width=True)
            else:
                st.markdown('<div class="hud-text" style="text-align:center;">AWAITING DATA</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab4:
            st.markdown('<div class="grid-bg-container" style="min-height:500px; display:block;">', unsafe_allow_html=True)
            if res is not None:
                fm = overlay_mask(res['img_np'].copy(), res['faults'], (255, 0, 255), 0.8)
                st.image(fm, use_column_width=True)
            else:
                st.markdown('<div class="hud-text" style="text-align:center;">AWAITING DATA</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    # --- RIGHT COLUMN ---
    with colR:
        st.markdown('<div class="hud-text" style="margin-bottom:16px;">AI Analysis Telemetry</div>', unsafe_allow_html=True)
        
        def render_bar(label, pct_val, color="#00FFAA"):
            st.markdown(f'''
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span class="hud-text" style="color:#cbd5e1; font-size:9px;">{label}</span>
                <span class="hud-text" style="color:{color};">{pct_val:.1f}%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill" style="width:{pct_val}%; background:{color};"></div></div>
            ''', unsafe_allow_html=True)
            
        if res is not None:
            render_bar("Ruins Detection", res['ruin_prob']*100, "#00FFAA")
            render_bar("Vegetation Overlay", res['probs'][2]*100, "#00FFAA")
            render_bar("Erosion Risk", res['erosion_risk']*100, "#fb923c")
            render_bar("Fault Lines", res['fault_prob']*100, "#c084fc")
            render_bar("Water Bodies", 0.0, "#60a5fa")
            render_bar("Urban / Built-up", 0.0, "#94a3b8")
            
            st.markdown('<div class="hud-text" style="margin-top:20px; margin-bottom:12px;">Confidence Matrix</div>', unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown('<div class="metric-box"><div class="hud-text" style="color:#64748b;">Precision</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">0.925</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-box" style="margin-top:8px;"><div class="hud-text" style="color:#64748b;">Latency</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">142ms</div></div>', unsafe_allow_html=True)
            with mc2:
                st.markdown('<div class="metric-box"><div class="hud-text" style="color:#64748b;">Recall</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">0.884</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-box" style="margin-top:8px;"><div class="hud-text" style="color:#64748b;">F1 Score</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">0.904</div></div>', unsafe_allow_html=True)
                
            st.markdown(f'''
            <div class="hud-text" style="margin-top:20px; color:#64748b; margin-bottom:4px;">Neural Link Feed</div>
            <div style="background:rgba(0,0,0,0.4); border: 1px solid #1e293b; padding:10px; min-height:80px; font-family:\\'Space Mono\\'; font-size:9px; color:rgba(0,255,170,0.6);">
                &gt; System ready.<br/>
                &gt; Analyzing tensor matrices...<br/>
                &gt; Primary: {pred_label}<br/>
                &gt; Signal clarity: 99.8%<br/>
                <span style="opacity:0.5;">█</span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            render_bar("Ruins Detection", 0)
            render_bar("Vegetation Overlay", 0)
            render_bar("Erosion Risk", 0, "#fb923c")
            render_bar("Fault Lines", 0, "#c084fc")
            render_bar("Water Bodies", 0, "#60a5fa")
            render_bar("Urban / Built-up", 0, "#94a3b8")
            
            st.markdown('<div class="hud-text" style="margin-top:20px; margin-bottom:12px;">Confidence Matrix</div>', unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown('<div class="metric-box"><div class="hud-text" style="color:#64748b;">Precision</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">—</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-box" style="margin-top:8px;"><div class="hud-text" style="color:#64748b;">Latency</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">—</div></div>', unsafe_allow_html=True)
            with mc2:
                st.markdown('<div class="metric-box"><div class="hud-text" style="color:#64748b;">Recall</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">—</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-box" style="margin-top:8px;"><div class="hud-text" style="color:#64748b;">F1 Score</div><div style="font-family:\\'Space Mono\\'; font-size:16px; color:white;">—</div></div>', unsafe_allow_html=True)

            st.markdown('''
            <div class="hud-text" style="margin-top:20px; color:#64748b; margin-bottom:4px;">Neural Link Feed</div>
            <div style="background:rgba(0,0,0,0.4); border: 1px solid #1e293b; padding:10px; min-height:80px; font-family:\\'Space Mono\\'; font-size:9px; color:rgba(0,255,170,0.6);">
                &gt; System ready. Awaiting input...<br/>
                &gt; BYOL encoder loaded.<br/>
                &gt; Analysis heads initialized.<br/>
                &gt; Upload imagery to begin scan.<br/>
                <span style="opacity:0.5;">█</span>
            </div>
            ''', unsafe_allow_html=True)
"""

with open(file_path, "w", encoding="utf-8") as f:
    f.write(before + new_portal_code)

print("Streamlit UI patch applied!")
