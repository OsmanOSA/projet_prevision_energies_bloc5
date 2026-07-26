import sys
sys.path.insert(0, "/app/streamlit_app")
exec(open("/app/streamlit_app/app_main.py", encoding="utf-8").read())
