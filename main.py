import time
import signal
import sys
from helpers import (
    start_crm_server,
    stop_crm_server,
    is_crm_running,
    start_ai_server,
    stop_ai_server,
    is_ai_running,
)

# Adjust these if you use a different host or port
CRM_URL = "http://127.0.0.1:8080"
AI_URL = "http://127.0.0.1:8501"


def start_servers():
    # CRM Server
    if not is_crm_running():
        print("Starting Lucrum CRM...")
        if start_crm_server():
            time.sleep(4)
            print(f"Lucrum CRM: 🟢 Online at {CRM_URL}")
        else:
            print("Lucrum CRM: ❌ Failed to start")
    else:
        print(f"Lucrum CRM: 🟢 Already running at {CRM_URL}")

    # AI Server
    if not is_ai_running():
        print("Starting Lucrum AI...")
        if start_ai_server():
            time.sleep(4)
            print(f"Lucrum AI: 🟢 Ready at {AI_URL}")
        else:
            print("Lucrum AI: ❌ Failed to start")
    else:
        print(f"Lucrum AI: 🟢 Already running at {AI_URL}")


def stop_servers():
    print("\nStopping all servers...")
    stop_crm_server()
    stop_ai_server()
    print("All servers stopped. Exiting.")


def main():
    start_servers()

    print("\nAccess URLs:")
    print(f"  🔗 Lucrum CRM: {CRM_URL}")
    print(f"  🔗 Lucrum AI:  {AI_URL}")
    print("\nPress Ctrl+C to shut down both servers and exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_servers()
        sys.exit(0)


if __name__ == "__main__":
    main()
