"""
ARCHAEOLIS Streamlit App - Root Deployment Entrypoint
"""
import os
import sys

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# Import and launch Streamlit App
if __name__ == "__main__":
    from dashboard.streamlit_app import *
else:
    from dashboard.streamlit_app import *
