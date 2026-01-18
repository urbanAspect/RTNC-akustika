import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import librosa

# Izračun SNR
def estimate_snr_db(path: str) -> float:
    # Ohrani izvorno vzorčno frekvenco in pretvori v mono za eno vrednost SNR na datoteko
    y, sr = librosa.load(path, sr=None, mono=True)

    if y is None or len(y) == 0:
        raise ValueError("Empty or unreadable audio.")

    # RMS za oceno 'signala' proti 'šumu'
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if rms.size == 0:
        raise ValueError("Could not compute RMS.")

    # Hevristika za SNR:
    # raven signala ~ 90th percentile RMS
    # raven šuma    ~ 10th percentile RMS
    signal = float(np.percentile(rms, 90))
    noise = float(np.percentile(rms, 10))

    eps = 1e-12
    snr_db = 20.0 * np.log10((signal + eps) / (noise + eps))
    return snr_db

# Grafični vmesnik
class SNRGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio SNR to TXT")
        self.geometry("720x520")
        self.minsize(680, 480)

        self.files = []

        self.output_path_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        pad = 10

        frm_files = ttk.LabelFrame(self, text="Files to measure")
        frm_files.pack(fill="both", expand=True, padx=pad, pady=(pad, 6))

        self.listbox = tk.Listbox(frm_files, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left", fill="both", expand=True, padx=(pad, 0), pady=pad)

        scrollbar = ttk.Scrollbar(frm_files, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="left", fill="y", padx=(6, pad), pady=pad)
        self.listbox.configure(yscrollcommand=scrollbar.set)

        btns = ttk.Frame(frm_files)
        btns.pack(side="right", fill="y", padx=(0, pad), pady=pad)

        ttk.Button(btns, text="Add files...", command=self.add_files).pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="Remove selected", command=self.remove_selected).pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="Clear", command=self.clear_files).pack(fill="x")

        frm_out = ttk.LabelFrame(self, text="Output TXT file path")
        frm_out.pack(fill="x", padx=pad, pady=6)

        entry = ttk.Entry(frm_out, textvariable=self.output_path_var)
        entry.pack(side="left", fill="x", expand=True, padx=(pad, 6), pady=pad)

        ttk.Button(frm_out, text="Browse...", command=self.browse_output).pack(side="left", padx=(0, pad), pady=pad)

        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", padx=pad, pady=6)

        ttk.Button(frm_actions, text="Compute SNR and save TXT", command=self.run).pack(side="left")

        self.log = tk.Text(self, height=10, wrap="word")
        self.log.pack(fill="both", expand=False, padx=pad, pady=(6, pad))
        self._log_line("Ready.")

    def _log_line(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.aiff *.aif"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert("end", p)

        self._log_line(f"Added {len(paths)} file(s).")

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            path = self.listbox.get(idx)
            self.listbox.delete(idx)
            if path in self.files:
                self.files.remove(path)
        self._log_line(f"Removed {len(sel)} file(s).")

    def clear_files(self):
        self.files.clear()
        self.listbox.delete(0, "end")
        self._log_line("Cleared file list.")

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Choose output TXT file",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.output_path_var.set(path)

    def run(self):
        out_path = self.output_path_var.get().strip()

        if not self.files:
            messagebox.showerror("Missing input", "Add at least one audio file.")
            return
        if not out_path:
            messagebox.showerror("Missing output", "Choose where to save the TXT file.")
            return

        self._log_line(f"Writing results to: {out_path}")
        errors = 0

        try:
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                for p in self.files:
                    base = os.path.basename(p)
                    try:
                        snr_db = estimate_snr_db(p)
                        f.write(f"{base} {snr_db:.2f}\n")
                        self._log_line(f"{base}: {snr_db:.2f} dB")
                    except Exception as e:
                        errors += 1
                        f.write(f"{base} ERROR\n")
                        self._log_line(f"{base}: ERROR ({e})")
        except Exception as e:
            messagebox.showerror("Write failed", f"Could not write output file:\n{e}")
            return

        if errors == 0:
            messagebox.showinfo("Done", "Computed SNR for all files and saved the TXT.")
        else:
            messagebox.showwarning("Done (with errors)", f"Saved TXT, but {errors} file(s) failed. See log.")


if __name__ == "__main__":
    SNRGui().mainloop()
