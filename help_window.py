import customtkinter as ctk
import webview
import webbrowser
import threading

from helpers import (
    start_crm_server, stop_crm_server, is_crm_running,
    start_ai_server, stop_ai_server, is_ai_running,
    create_button, create_label
)

class HelpWindow(ctk.CTkToplevel):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.title("Help & Troubleshooting")
        self.geometry("450x400")
        self.configure(bg="#000000")
        self.attributes("-topmost", True)
        self.resizable(False, False)
        self.grab_set()

        # Top section frame
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=10, anchor="n")

        home_label = create_label(top_frame, "← Home", 14, color="#ffffff", side="left")
        home_label.bind("<Button-1>", lambda e: self.destroy())
        home_label.configure(cursor="hand2")
        create_label(top_frame, "? Help & Troubleshooting", 14, color="#ffffff", side="right")

        crm_frame = ctk.CTkFrame(self, fg_color="transparent")
        crm_frame.pack(padx=20, pady=5, anchor="w")

        # Server management
        server_mgmt_frame = ctk.CTkFrame(crm_frame, fg_color="transparent")
        server_mgmt_frame.pack(fill="x")

        crm_section = ctk.CTkFrame(server_mgmt_frame, fg_color="transparent")
        crm_section.pack(side="left", padx=(0, 10))
        create_label(crm_section, "CRM Server Management", 14, color="#ffffff", pady=5)
        self.crm_toggle_btn = create_button(crm_section, "", self.toggle_crm, width=200)

        ai_section = ctk.CTkFrame(server_mgmt_frame, fg_color="transparent")
        ai_section.pack(side="left")
        create_label(ai_section, "AI Server Management", 14, color="#ffffff", pady=5)
        self.ai_toggle_btn = create_button(ai_section, "", self.toggle_ai, width=200)

        # Access buttons
        access_frame = ctk.CTkFrame(self, fg_color="transparent")
        access_frame.pack(padx=20, pady=5, fill="x")

        crm_access_section = ctk.CTkFrame(access_frame, fg_color="transparent")
        crm_access_section.pack(side="left", padx=(0, 10))
        create_label(crm_access_section, "CRM Access", 14, color="#ffffff", pady=10)
        self.crm_browser_btn = create_button(crm_access_section, "Open CRM in Browser", lambda: webbrowser.open("http://localhost:8080"), width=200)
        self.crm_window_btn = create_button(crm_access_section, "Open CRM in Window", self.open_crm_window, width=200)

        ai_access_section = ctk.CTkFrame(access_frame, fg_color="transparent")
        ai_access_section.pack(side="left")
        create_label(ai_access_section, "AI Access", 14, color="#ffffff", pady=10)
        self.ai_browser_btn = create_button(ai_access_section, "Open AI in Browser", lambda: webbrowser.open("http://localhost:8501"), width=200)
        self.ai_window_btn = create_button(ai_access_section, "Open AI in Window", self.open_ai_window, width=200)

        self.update_toggle_buttons()

    def update_toggle_buttons(self):
        # CRM toggle
        if is_crm_running():
            self.crm_toggle_btn.configure(text="Stop CRM Server", fg_color="#D94E4E")
            self.crm_browser_btn.configure(state=ctk.NORMAL, fg_color="#D2A458", text_color="#000000")
            self.crm_window_btn.configure(state=ctk.NORMAL, fg_color="#D2A458", text_color="#000000")
        else:
            self.crm_toggle_btn.configure(text="Start CRM Server", fg_color="#4ED96A")
            self.crm_browser_btn.configure(state=ctk.DISABLED, fg_color="#333333", text_color="#777777")
            self.crm_window_btn.configure(state=ctk.DISABLED, fg_color="#333333", text_color="#777777")

        # AI toggle
        if is_ai_running():
            self.ai_toggle_btn.configure(text="Stop AI Server", fg_color="#D94E4E")
            self.ai_browser_btn.configure(state=ctk.NORMAL, fg_color="#D2A458", text_color="#000000")
            self.ai_window_btn.configure(state=ctk.NORMAL, fg_color="#D2A458", text_color="#000000")
        else:
            self.ai_toggle_btn.configure(text="Start AI Server", fg_color="#4ED96A")
            self.ai_browser_btn.configure(state=ctk.DISABLED, fg_color="#333333", text_color="#777777")
            self.ai_window_btn.configure(state=ctk.DISABLED, fg_color="#333333", text_color="#777777")

    def toggle_crm(self):
        self.crm_toggle_btn.configure(text="Processing...", fg_color="#777777", state=ctk.DISABLED)
        self.update_idletasks()

        def task():
            if is_crm_running():
                stop_crm_server()
                status = ("Lucrum CRM: Stopped", "#D94E4E")
            else:
                start_crm_server()
                status = ("Lucrum CRM: 🟢 Online and ready", "#4ED96A")

            self.after(1000, lambda: self.main_app.update_crm_status(*status))
            self.after(1000, self.update_toggle_buttons)
            self.after(1000, lambda: self.crm_toggle_btn.configure(state=ctk.NORMAL))

        threading.Thread(target=task, daemon=True).start()

    def toggle_ai(self):
        self.ai_toggle_btn.configure(text="Processing...", fg_color="#777777", state=ctk.DISABLED)
        self.update_idletasks()

        def task():
            if is_ai_running():
                stop_ai_server()
                status = ("Lucrum AI: Stopped", "#D94E4E")
            else:
                start_ai_server()
                status = ("Lucrum AI: 🟢 Ready", "#4ED96A")

            self.after(1000, lambda: self.main_app.update_ai_status(*status))
            self.after(1000, self.update_toggle_buttons)
            self.after(1000, lambda: self.ai_toggle_btn.configure(state=ctk.NORMAL))

        threading.Thread(target=task, daemon=True).start()

    def open_crm_window(self):
        webview.create_window("EspoCRM", "http://localhost:8080")
        webview.start()

    def open_ai_window(self):
        webview.create_window("Lucrum AI", "http://localhost:8501")
        webview.start()