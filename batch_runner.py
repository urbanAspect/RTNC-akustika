import os
import sys
import subprocess
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# --- STANDALONE WORKER FUNCTION ---
def run_single_file(file_path, script, model, out_dir, idx):
    fname = os.path.basename(file_path)
    temp_in = os.path.join(out_dir, f"proc_{idx}_in.wav")
    temp_out = os.path.join(out_dir, f"proc_{idx}_out.wav")
    final_out = os.path.join(out_dir, f"Denoised_{os.path.splitext(fname)[0]}.wav")
    
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

        # 1. FFmpeg to 16k Mono WAV
        subprocess.run(['ffmpeg', '-y', '-i', file_path, '-ar', '16000', '-ac', '1', temp_in], 
                       check=True, capture_output=True)
        
        # 2. Run Inference Script
        cmd = [sys.executable, script, "-m", model, "-if", temp_in, "-of", temp_out]
        subprocess.run(cmd, check=True, capture_output=True)
        
        if os.path.exists(temp_out):
            if os.path.exists(final_out): os.remove(final_out)
            os.rename(temp_out, final_out)
            return f"SUCCESS: {fname}"
    except Exception as e:
        return f"ERROR: {fname} -> {str(e)}"
    finally:
        if os.path.exists(temp_in): os.remove(temp_in)
        if os.path.exists(temp_out): os.remove(temp_out)

# --- GUI CLASS ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class NoiseSuppressorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Batch denoiser")
        self.geometry("950x800")
        
        self.physical_cores = psutil.cpu_count(logical=False)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.suppressor_script = os.path.join(self.script_dir, "noise_suppressor.py")
        self.model_path = os.path.join(self.script_dir, "intel/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.xml")
        self.output_dir = os.path.join(self.script_dir, "denoised_output")
        self.selected_files = []

        # --- UI Setup ---
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="Batch denoiser", font=("Segoe UI", 24, "bold")).grid(row=0, column=0, pady=20)

        self.config_frame = ctk.CTkFrame(self)
        self.config_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.config_frame.grid_columnconfigure(1, weight=1)

        # Path Inputs with Browse Buttons
        self.script_entry = self.add_setting("Script Path:", self.suppressor_script, 0, self.browse_script)
        self.model_entry = self.add_setting("Model Path:", self.model_path, 1, self.browse_model)
        self.output_entry = self.add_setting("Output Folder:", self.output_dir, 2, self.browse_output)

        # Thread Input
        thread_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        thread_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        ctk.CTkLabel(thread_frame, text="Parallel Processes:").pack(side="left")
        self.thread_input = ctk.CTkEntry(thread_frame, width=60)
        self.thread_input.pack(side="left", padx=10)
        self.thread_input.insert(0, str(self.physical_cores))

        self.file_btn = ctk.CTkButton(self, text="Add Audio Files", command=self.select_files)
        self.file_btn.grid(row=2, column=0, pady=10)

        self.textbox = ctk.CTkTextbox(self, width=900, height=300)
        self.textbox.grid(row=3, column=0, padx=20, pady=10)

        self.progress_bar = ctk.CTkProgressBar(self, width=900)
        self.progress_bar.grid(row=4, column=0, pady=10)
        self.progress_bar.set(0)

        self.run_btn = ctk.CTkButton(self, text="Start Processing", command=self.start, fg_color="#1a73e8", height=45)
        self.run_btn.grid(row=5, column=0, pady=20)

    def add_setting(self, label, default, row, browse_command):
        ctk.CTkLabel(self.config_frame, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        e = ctk.CTkEntry(self.config_frame, width=500)
        e.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        e.insert(0, default)
        ctk.CTkButton(self.config_frame, text="Browse", width=80, command=browse_command).grid(row=row, column=2, padx=10, pady=5)
        return e

    # --- BROWSE HANDLERS ---
    def browse_script(self):
        path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
        if path:
            self.script_entry.delete(0, "end")
            self.script_entry.insert(0, path)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("OpenVINO XML", "*.xml")])
        if path:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
        if files:
            self.selected_files = list(files)
            self.textbox.delete("0.0", "end")
            self.textbox.insert("end", f"Queue: {len(files)} files.")

    def start(self):
        if not self.selected_files:
            messagebox.showwarning("Warning", "No files selected.")
            return
        threading.Thread(target=self.run_batch, daemon=True).start()

    def run_batch(self):
        script = self.script_entry.get()
        model = self.model_entry.get()
        out_dir = self.output_entry.get()
        try:
            workers = int(self.thread_input.get())
        except:
            workers = self.physical_cores

        os.makedirs(out_dir, exist_ok=True)
        self.run_btn.configure(state="disabled")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_single_file, f, script, model, out_dir, i): f for i, f in enumerate(self.selected_files)}
            for i, future in enumerate(as_completed(futures)):
                self.textbox.insert("end", f"\n{future.result()}")
                self.textbox.see("end")
                self.progress_bar.set((i + 1) / len(self.selected_files))

        self.run_btn.configure(state="normal")
        messagebox.showinfo("Done", "Processing Finished")

if __name__ == "__main__":
    import multiprocessing
    app = NoiseSuppressorGUI()
    app.mainloop()