import customtkinter as ctk
import subprocess
import psutil
import os
import logging
import signal

# ---------- Paths ----------
# system_dir = os.path.join(os.getcwd(), "system")
# os.chdir(system_dir)
ai_path = "lucrum_ai.py"
project_root = os.path.dirname(os.path.abspath(__file__))
compose_path = os.path.join(project_root, "docker", "docker-compose.yml")

# ---------- UI Helpers ----------

def create_label(parent, text, size, color="#ffffff", pady=5, side=None):
    label = ctk.CTkLabel(parent, text=text, font=("Segoe UI", size), text_color=color)
    if side:
        label.pack(side=side, pady=pady)
    else:
        label.pack(pady=pady)
    return label

def create_button(parent, text, command, fg="#D2A458", width=240):
    button = ctk.CTkButton(
        parent, text=text, command=command,
        fg_color=fg, text_color="#000000",
        width=width, height=40
    )
    button.pack(pady=5)
    return button

# ---------- Service Helpers ----------

def start_crm_server():
    try:
        subprocess.run(
            ["docker-compose", "-f", compose_path, "up", "-d"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info("CRM Docker container started.")
        return True
    except Exception as e:
        logging.error(f"Error starting CRM Docker container: {e}")
        return False

def stop_crm_server():
    try:
        subprocess.run(
            ["docker-compose", "-f", compose_path, "down"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info("CRM Docker container stopped.")
        return True
    except Exception as e:
        logging.error(f"Error stopping CRM Docker container: {e}")
        return False

def is_crm_running():
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=espocrm", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        # If the container name appears in the output, it's running
        return "espocrm" in result.stdout
    except Exception as e:
        logging.error(f"Error checking CRM Docker container: {e}")
        return False

def start_ai_server():
    try:
        subprocess.Popen(
            ["streamlit", "run", ai_path, "--server.headless", "true", "--browser.serverAddress", "127.0.0.1"],
            #cwd=system_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info("AI server started.")
        return True
    except Exception as e:
        logging.error(f"Error starting AI server: {e}")
        return False

def stop_ai_server():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if "streamlit" in " ".join(proc.info.get('cmdline', [])):
            try:
                os.kill(proc.info['pid'], signal.SIGTERM)
                logging.info("AI server stopped.")
            except Exception as e:
                logging.error(f"Error stopping AI server: {e}")

def is_ai_running():
    for proc in psutil.process_iter(['name', 'cmdline']):
        if "streamlit" in " ".join(proc.info.get('cmdline', [])):
            return True
    return False
