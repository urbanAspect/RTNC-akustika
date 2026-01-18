import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import threading

class AudioCombinerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Combiner")
        self.root.geometry("460x320")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('vista')

        # Spremenljivke
        self.input1_files = []
        self.input2_files = []
        self.input1_display = tk.StringVar(value="No files selected")
        self.input2_display = tk.StringVar(value="No files selected")
        self.output_folder = tk.StringVar()
        
        self.vol1_val = tk.StringVar(value="100")
        self.vol2_val = tk.StringVar(value="100")
        self.loop_shorter = tk.BooleanVar(value=False)
        self.crop_to_in2 = tk.BooleanVar(value=False)

        self.main_frame = ttk.Frame(root, padding="20 10 20 20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.build_ui()

    def build_ui(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.main_frame, text="Audio Combiner", font=("Segoe UI", 10, "bold"), foreground="black").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,10))
        ttk.Button(self.main_frame, text="Help", command=self.show_help).grid(row=0, column=2, sticky="e")

        # Input 1
        ttk.Label(self.main_frame, text="Input 1 (Main):").grid(row=1, column=0, sticky="w")
        ttk.Label(self.main_frame, text="Vol %:").grid(row=1, column=1, sticky="w")
        ttk.Entry(self.main_frame, textvariable=self.input1_display, width=45, state='readonly').grid(row=2, column=0, padx=(0, 10), pady=(0, 10))
        ttk.Spinbox(self.main_frame, from_=0, to=200, textvariable=self.vol1_val, width=5).grid(row=2, column=1, padx=(0, 10), pady=(0, 10))
        ttk.Button(self.main_frame, text="Browse...", command=self.browse_in1).grid(row=2, column=2, pady=(0, 10))

        # Input 2
        ttk.Label(self.main_frame, text="Input 2 (Overlay):").grid(row=3, column=0, sticky="w")
        ttk.Label(self.main_frame, text="Vol %:").grid(row=3, column=1, sticky="w")
        ttk.Entry(self.main_frame, textvariable=self.input2_display, width=45, state='readonly').grid(row=4, column=0, padx=(0, 10), pady=(0, 10))
        ttk.Spinbox(self.main_frame, from_=0, to=200, textvariable=self.vol2_val, width=5).grid(row=4, column=1, padx=(0, 10), pady=(0, 10))
        ttk.Button(self.main_frame, text="Browse...", command=self.browse_in2).grid(row=4, column=2, pady=(0, 10))

        # Output
        ttk.Label(self.main_frame, text="Output Folder:").grid(row=5, column=0, sticky="w")
        ttk.Entry(self.main_frame, textvariable=self.output_folder, width=45).grid(row=6, column=0, padx=(0, 10), pady=(0, 10))
        ttk.Button(self.main_frame, text="Select Folder", command=self.browse_folder).grid(row=6, column=2, pady=(0, 10))

        # Options
        opt_frame = ttk.Frame(self.main_frame)
        opt_frame.grid(row=7, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Checkbutton(opt_frame, text="Loop Input 2", variable=self.loop_shorter).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(opt_frame, text="Crop to Input 2", variable=self.crop_to_in2).pack(side=tk.LEFT, padx=5)

        # Buttons
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)
        self.combine_btn = ttk.Button(btn_frame, text="START PROCESSING", command=self.start_processing)
        self.combine_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.status_label = ttk.Label(self.main_frame, text="Ready", foreground="gray")
        self.status_label.grid(row=9, column=0, columnspan=3)

    def browse_in1(self):
        f = filedialog.askopenfilenames(filetypes=[("Media", "*.mp3 *.wav *.aac *.mp4 *.mkv *.avi *.mov *.flac")])
        if f: self.input1_files = list(f); self.input1_display.set(f"{len(f)} files selected")

    def browse_in2(self):
        f = filedialog.askopenfilenames(filetypes=[("Media", "*.mp3 *.wav *.aac *.mp4 *.mkv *.avi *.mov *.flac")])
        if f: self.input2_files = list(f); self.input2_display.set(f"{len(f)} files selected")

    def browse_folder(self):
        f = filedialog.askdirectory(); self.output_folder.set(f) if f else None

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("User Instructions")
        help_window.geometry("500x420")
        help_text = (
            "User Instructions\n\n"
            "1. SELECT INPUTS:\n"
            "   Select multiple files for both Input 1 and Input 2.\n\n"
            "2. ADJUST VOLUME:\n"
            "   100 = Original volume\n   0 = Muted\n   200 = Double\n\n"
            "3. LOOPING OPTION:\n"
            "   Loops Input 2 repeatedly until Input 1 finishes.\n\n"
            "4. CROP TO INPUT 2:\n"
            "   Forces the output to end exactly when Input 2 ends.\n\n"
            "5. OUTPUT:\n"
            "   Files are saved as 'Input1Name_X_Input2Name.mp3'."
        )
        tk.Label(help_window, text=help_text, justify=tk.LEFT, padx=20, pady=20, font=("Segoe UI", 10)).pack()

    def start_processing(self):
        if not self.input1_files or not self.input2_files or not self.output_folder.get():
            messagebox.showwarning("Warning", "Selection missing."); return
        self.combine_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.process_logic).start()

    # Združevalna logika
    def process_logic(self):
        out_dir = self.output_folder.get()
        v1, v2 = int(self.vol1_val.get())/100.0, int(self.vol2_val.get())/100.0
        total = len(self.input1_files) * len(self.input2_files)
        current = 0

        for f1 in self.input1_files:
            for f2 in self.input2_files:
                current += 1
                out_name = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(f1))[0]}_X_{os.path.splitext(os.path.basename(f2))[0]}.mp3")
                self.root.after(0, lambda c=current: self.status_label.config(text=f"Task {c}/{total}..."))

                args = ["-i", f1]
                if self.loop_shorter.get(): args += ["-stream_loop", "-1"]
                args += ["-i", f2]

                # Nastavitve trajanja
                if self.crop_to_in2.get():
                    dur = "shortest"
                elif self.loop_shorter.get():
                    dur = "first"
                else:
                    dur = "longest"

                # Filter kompleks
                filter_complex = (
                    f"[0:a]aresample=44100,volume={v1}[a1]; "
                    f"[1:a]aresample=44100,volume={v2}[a2]; "
                    f"[a1][a2]amix=inputs=2:duration={dur}:dropout_transition=0"
                )

                # Združevanje s ffmpeg
                cmd = ["ffmpeg"] + args + ["-filter_complex", filter_complex, "-ac", "2", "-y", out_name]

                sinfo = subprocess.STARTUPINFO()
                sinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                subprocess.run(cmd, capture_output=True, startupinfo=sinfo)

        self.root.after(0, self.finish)

    def finish(self):
        self.combine_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Done!", foreground="green")
        messagebox.showinfo("Success", "All files combined successfully.")

if __name__ == "__main__":
    root = tk.Tk(); app = AudioCombinerApp(root); root.mainloop()