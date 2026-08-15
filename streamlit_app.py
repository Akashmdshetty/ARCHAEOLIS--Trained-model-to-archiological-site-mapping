"""
ARCHAEOLIS Streamlit App - Root Deployment Entrypoint
"""
import os
import sys

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Execute the dashboard application script directly on every Streamlit rerun
dashboard_script = os.path.join(ROOT_DIR, "dashboard", "streamlit_app.py")
with open(dashboard_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), dashboard_script, "exec")
    exec(code, globals())
