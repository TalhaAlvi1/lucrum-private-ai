import webview

# CRM URL (local or remote)
CRM_URL = "http://localhost:8080"  # Change this to your actual CRM URL

# Create and start a window with the CRM
webview.create_window("LucrumCRM", CRM_URL, width=1080, height=720,)
webview.start()
