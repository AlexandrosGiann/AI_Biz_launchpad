import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
import os
import re
import platform
import tempfile
import shutil
import json
import requests
from urllib.parse import urlencode
from pathlib import Path as _Path

# --- Optional deps are handled inside functions where needed (Pillow/pydub) ---
try:
    import ollama
except Exception as e:
    raise SystemExit("Απαιτείται το 'ollama' (pip install ollama). " + str(e))
try:
    import pyttsx3
except Exception as e:
    raise SystemExit("Απαιτείται το 'pyttsx3' (pip install pyttsx3). " + str(e))

APP_TITLE = "AI Content Creation"
MODELS_FILE = "models.txt"
API_KEYS_FILE = "api_keys.json"


# ---------------- TTS Worker (σταθερό, single thread) ----------------
class TTSWorker(threading.Thread):
    def __init__(self, notify_queue: queue.Queue):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.notify_q = notify_queue
        self._stop = threading.Event()
        self.engine = None
        self.voices = []

    def run(self):
        self.engine = pyttsx3.init()
        try:
            self.voices = self.engine.getProperty('voices') or []
        except Exception:
            self.voices = []
        while not self._stop.is_set():
            try:
                cmd = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                op = cmd.get('op')
                if op == 'set':
                    vid = cmd.get('voice_id')
                    if vid:
                        try:
                            self.engine.setProperty('voice', vid)
                        except Exception as e:
                            self._notify('tts_error', f"Voice set error: {e}")
                    rate = cmd.get('rate')
                    if isinstance(rate, int):
                        try:
                            self.engine.setProperty('rate', rate)
                        except Exception as e:
                            self._notify('tts_error', f"Rate set error: {e}")
                    vol = cmd.get('volume')
                    if isinstance(vol, (int, float)):
                        try:
                            self.engine.setProperty('volume', float(vol))
                        except Exception as e:
                            self._notify('tts_error', f"Volume set error: {e}")
                elif op == 'set_by_index':
                    idx = cmd.get('index')
                    try:
                        if isinstance(idx, int) and self.voices and 0 <= idx < len(self.voices):
                            self.engine.setProperty('voice', self.voices[idx].id)
                            self._notify('tts_done', f"Voice set by index {idx}.")
                        else:
                            self._notify('tts_error', f"Voice index {idx} not available.")
                    except Exception as e:
                        self._notify('tts_error', f"Voice set_by_index error: {e}")
                elif op == 'preview':
                    text = cmd.get('text', '')
                    if text:
                        self.engine.stop()
                        self.engine.say(text)
                        self.engine.runAndWait()
                        self._notify('tts_done', "Preview finished.")
                elif op == 'export':
                    tasks = cmd.get('tasks', [])
                    if tasks:
                        self.engine.stop()
                        for txt, path in tasks:
                            self.engine.save_to_file(txt, path)
                        self.engine.runAndWait()
                        self._notify('tts_done', f"Exported {len(tasks)} file(s).")
                elif op == 'export_and_convert_mp3':
                    tasks_mp3 = cmd.get('tasks', [])
                    if tasks_mp3:
                        self.engine.stop()
                        tmpdir = tempfile.mkdtemp(prefix="tts_wav_")
                        wav_paths = []
                        try:
                            for txt, mp3_path in tasks_mp3:
                                base = os.path.splitext(os.path.basename(mp3_path))[0]
                                wav_path = os.path.join(tmpdir, f"{base}.wav")
                                wav_paths.append((wav_path, mp3_path))
                                self.engine.save_to_file(txt, wav_path)
                            self.engine.runAndWait()
                            try:
                                from pydub import AudioSegment
                            except Exception as e:
                                self._notify('tts_error', f"Λείπει το pydub/ffmpeg για MP3: {e}")
                                return
                            converted = 0
                            for wav_path, mp3_path in wav_paths:
                                try:
                                    AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
                                    converted += 1
                                except Exception as e:
                                    self._notify('tts_error', f"Μετατροπή σε MP3 απέτυχε για {mp3_path}: {e}")
                            self._notify('tts_done', f"Exported {converted} MP3 file(s).")
                        finally:
                            try:
                                shutil.rmtree(tmpdir, ignore_errors=True)
                            except Exception:
                                pass
                elif op == 'stop':
                    self.engine.stop()
                elif op == 'shutdown':
                    self.engine.stop()
                    break
            finally:
                self.q.task_done()

    def enqueue(self, cmd):
        self.q.put(cmd)

    def shutdown(self):
        self.enqueue({'op': 'shutdown'})
        self._stop.set()

    def _notify(self, kind, message):
        try:
            self.notify_q.put((kind, message))
        except Exception:
            pass


# ---------------- Main App ----------------
class AiContentApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1140x860")

        self.stream_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.last_generated_text = ""

        self.tts = TTSWorker(self.stream_queue)
        self.tts.start()

        # API keys
        self.api_keys = {"pexels": "", "pixabay": "", "unsplash": ""}
        self.load_api_keys()

        # -------- Options --------
        options = ttk.Frame(root, padding=10)
        options.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(options, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(options, textvariable=self.model_var, state="readonly", width=30)
        self.model_combo.grid(row=0, column=1, sticky="w")
        self.reload_btn = ttk.Button(options, text="Reload models.txt", command=self.load_models)
        self.reload_btn.grid(row=0, column=2, padx=(8, 0))

        ttk.Label(options, text="Prompt:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.prompt_var = tk.StringVar(value="Create a script for a video about ")
        self.prompt_entry = ttk.Entry(options, textvariable=self.prompt_var, width=80)
        self.prompt_entry.grid(row=1, column=1, columnspan=2, sticky="we", pady=(10, 0))

        # -------- Voice settings --------
        tts_box = ttk.LabelFrame(root, text="Voice Settings", padding=10)
        tts_box.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))

        ttk.Label(tts_box, text="Voice:").grid(row=0, column=0, sticky="w")
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(tts_box, textvariable=self.voice_var, state="readonly", width=54)
        self.voice_combo.grid(row=0, column=1, sticky="we", padx=(6, 12))
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_change)

        ttk.Label(tts_box, text="Rate:").grid(row=0, column=2, sticky="e")
        self.rate_var = tk.IntVar(value=200)
        self.rate_slider = ttk.Scale(tts_box, from_=100, to=300, orient="horizontal", command=self.on_rate_slide)
        self.rate_slider.set(self.rate_var.get())
        self.rate_slider.grid(row=0, column=3, sticky="we", padx=6)

        ttk.Label(tts_box, text="Volume:").grid(row=0, column=4, sticky="e")
        self.volume_var = tk.DoubleVar(value=1.0)
        self.volume_slider = ttk.Scale(tts_box, from_=0.1, to=1.0, orient="horizontal", command=self.on_volume_slide)
        self.volume_slider.set(self.volume_var.get())
        self.volume_slider.grid(row=0, column=5, sticky="we", padx=6)

        tts_box.columnconfigure(1, weight=1)
        tts_box.columnconfigure(3, weight=1)
        tts_box.columnconfigure(5, weight=1)

        # -------- Action buttons --------
        buttons = ttk.Frame(root, padding=(10, 0, 10, 10))
        buttons.pack(side=tk.TOP, fill=tk.X)
        self.run_btn = ttk.Button(buttons, text="Submit", command=self.start_generation)
        self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_generation, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.save_btn = ttk.Button(buttons, text="Save Output…", command=self.save_output)
        self.save_btn.pack(side=tk.RIGHT)

        # -------- TTS buttons --------
        tts_btns = ttk.Frame(root, padding=(10, 0, 10, 10))
        tts_btns.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.preview_btn = ttk.Button(tts_btns, text="Preview (quotes only)", command=self.preview_tts)
        self.preview_btn.pack(side=tk.LEFT)
        self.stop_tts_btn = ttk.Button(tts_btns, text="Stop Speech", command=self.stop_tts)
        self.stop_tts_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(tts_btns, text="Export prefix:").pack(side=tk.LEFT, padx=(16, 4))
        self.prefix_var = tk.StringVar(value="script_part")
        ttk.Entry(tts_btns, textvariable=self.prefix_var, width=20).pack(side=tk.LEFT)

        ttk.Label(tts_btns, text="Format:").pack(side=tk.LEFT, padx=(12, 4))
        self.audio_fmt_var = tk.StringVar(value="wav")
        ttk.Combobox(tts_btns, textvariable=self.audio_fmt_var, state="readonly", width=8,
                     values=["wav", "aiff", "mp3"]).pack(side=tk.LEFT)

        self.auto_audio_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tts_btns, text="Auto-export after generation", variable=self.auto_audio_var).pack(side=tk.RIGHT)
        self.export_btn = ttk.Button(tts_btns, text="Export Quotes to Audio…", command=self.export_audio_quotes)
        self.export_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # -------- Output area --------
        text_frame = ttk.Frame(root, padding=10)
        text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(text_frame, wrap="word", height=18)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.output_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=yscroll.set)

        # -------- Stock Images --------
        stock = ttk.LabelFrame(root, text="Stock Images", padding=10)
        stock.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 8))

        ttk.Label(stock, text="Provider:").grid(row=0, column=0, sticky="w")
        self.provider_var = tk.StringVar(value="Google Images (scraper)")
        self.provider_combo = ttk.Combobox(
            stock, textvariable=self.provider_var, state="readonly", width=24,
            values=["Google Images (scraper)", "Pixabay", "Pexels", "Unsplash (images)"]
        )
        self.provider_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(stock, text="Query:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.stock_query_var = tk.StringVar()
        self.stock_query_entry = ttk.Entry(stock, textvariable=self.stock_query_var, width=60)
        self.stock_query_entry.grid(row=1, column=1, columnspan=2, sticky="we", pady=(8, 0))

        self.use_quote_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(stock, text="Use selected quote as query", variable=self.use_quote_var)\
            .grid(row=1, column=3, sticky="w", padx=(8, 0))

        ttk.Label(stock, text="Max images:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.max_items_var = tk.IntVar(value=10)
        ttk.Spinbox(stock, from_=1, to=100, textvariable=self.max_items_var, width=6)\
            .grid(row=2, column=1, sticky="w", pady=(8, 0))

        self.keys_btn = ttk.Button(stock, text="API Keys…", command=self._open_keys_dialog)
        self.keys_btn.grid(row=2, column=2, sticky="w", pady=(8, 0))

        self.download_btn = ttk.Button(stock, text="Download Images…", command=self._download_images)
        self.download_btn.grid(row=2, column=3, sticky="e", pady=(8, 0))

        ttk.Label(stock, text="Per-quote images:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.per_quote_var = tk.IntVar(value=2)
        ttk.Spinbox(stock, from_=1, to=10, textvariable=self.per_quote_var, width=6)\
            .grid(row=3, column=1, sticky="w", pady=(6, 0))

        self.auto_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stock, text="Auto-download images after Submit (by topic)", variable=self.auto_images_var)\
            .grid(row=3, column=2, sticky="w", pady=(6, 0))

        self.auto_per_quote_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stock, text="Auto-download per quote after generation", variable=self.auto_per_quote_var)\
            .grid(row=3, column=3, sticky="w", pady=(6, 0))

        self.subfolders_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stock, text="Create subfolders per quote", variable=self.subfolders_var)\
            .grid(row=4, column=1, sticky="w", pady=(6, 0))

        self.convert_jpg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stock, text="Convert to JPG (compatibility)", variable=self.convert_jpg_var)\
            .grid(row=4, column=2, sticky="w", pady=(6, 0))

        stock.columnconfigure(1, weight=1)
        stock.columnconfigure(2, weight=1)
        stock.columnconfigure(3, weight=1)

        # -------- Status --------
        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(root, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self.load_models()
        self.load_voices()
        self.root.after(50, self._drain_queue)

    # ---------------- Models ----------------
    def load_models(self):
        models = []
        try:
            with open(MODELS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    models.append(s)
        except FileNotFoundError:
            pass
        if not models:
            models = ["llama3.1:8b"]
        self.model_combo["values"] = models
        if not self.model_var.get():
            self.model_var.set(models[0])
        self.status_var.set(f"Loaded {len(models)} model(s).")

    # ---------------- Voices ----------------
    def load_voices(self):
        voices = []
        for driver in (None, 'sapi5', 'nsss', 'espeak', 'espeak-ng'):
            try:
                tmp = pyttsx3.init(driverName=driver) if driver else pyttsx3.init()
                voices = tmp.getProperty("voices") or []
                if voices:
                    break
            except Exception:
                continue
        self._voices = voices

        items = []
        if voices:
            items.append(f"voice1 — {voices[0].name}")
            if len(voices) > 2:
                items.append(f"voice2 — {voices[1].name}")
            items.append("────────")
            for v in voices:
                items.append(f"{(v.name or 'Voice')} — {v.id}")
        else:
            items = ["voice1 (engine[0])", "voice2 (engine[1])"]

        self.voice_combo["values"] = items

        preferred_index = 0
        if voices:
            for i, v in enumerate(voices):
                name = (v.name or "").lower()
                vid = (v.id or "").lower()
                if "zira" in name or "zira" in vid:
                    preferred_index = 3 + i
                    break
        self.voice_combo.current(preferred_index)
        self.on_voice_change()

    def on_voice_change(self, *_):
        sel_text = self.voice_combo.get()
        if sel_text.startswith("voice1"):
            self.status_var.set("Selected voice1 → engine[0]")
            self.tts.enqueue({'op': 'set_by_index', 'index': 0})
            return
        if sel_text.startswith("voice2"):
            self.status_var.set("Selected voice2 → engine[1]")
            self.tts.enqueue({'op': 'set_by_index', 'index': 1})
            return

        try:
            items = list(self.voice_combo["values"])
            sep_index = items.index("────────")
        except Exception:
            sep_index = -1
        sel = self.voice_combo.current()
        vidx = 0
        if sep_index != -1 and sel > sep_index:
            vidx = sel - (sep_index + 1)
        try:
            v = self._voices[vidx]
            self.status_var.set(f"Selected voice: {v.name} — {v.id}")
            self.tts.enqueue({'op': 'set', 'voice_id': v.id})
        except Exception:
            self.status_var.set("Fallback to voice1 (engine[0])")
            self.tts.enqueue({'op': 'set_by_index', 'index': 0})

    def on_rate_slide(self, val):
        try:
            rate = int(float(val))
            self.tts.enqueue({'op': 'set', 'rate': rate})
        except Exception:
            pass

    def on_volume_slide(self, val):
        try:
            vol = float(val)
            self.tts.enqueue({'op': 'set', 'volume': vol})
        except Exception:
            pass

    # ---------------- Generation ----------------
    def start_generation(self):
        model = self.model_var.get().strip()
        prompt = self.prompt_var.get().strip()
        if not model:
            messagebox.showwarning(APP_TITLE, "Παρακαλώ επιλέξτε μοντέλο.")
            return
        if not prompt:
            messagebox.showwarning(APP_TITLE, "Το prompt δεν μπορεί να είναι κενό.")
            return

        self.output_text.delete("1.0", tk.END)
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set(f"Generating with '{model}'…")
        self.stop_event.clear()

        self.worker = threading.Thread(target=self._worker_generate, args=(model, prompt), daemon=True)
        self.worker.start()

        # Auto-download by topic: ΡΩΤΑ ΦΑΚΕΛΟ στο main thread, μετά ξεκίνα background
        try:
            if self.auto_images_var.get():
                self.root.after(0, lambda: self._auto_download_images_from_prompt_ui(prompt))
        except Exception:
            pass

    def stop_generation(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("Stopping…")

    def _worker_generate(self, model, prompt):
        t0 = time.time()
        try:
            system_msg = (
                "You are a helpful writing assistant. "
                "Write a video script where ONLY the voice-over narration lines are enclosed in DOUBLE QUOTES. "
                "Use headings for scenes without quotes (e.g., Scene 1:, B-roll:, SFX:). "
                "Every narratable line MUST be placed inside double quotes (\"). "
                "Keep sentences short and natural for TTS."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            full_text_parts = []
            for part in ollama.chat(model=model, messages=messages, stream=True):
                if self.stop_event.is_set():
                    break
                delta = part.get("message", {}).get("content", "")
                if delta:
                    full_text_parts.append(delta)
                    self.stream_queue.put(("text", delta))
            total = "".join(full_text_parts)
            self.last_generated_text = total
            self.stream_queue.put(("done", time.time() - t0))

            # Auto per-quote: ΡΩΤΑ ΦΑΚΕΛΟ στο main thread, μετά background
            if self.auto_per_quote_var.get() and total.strip():
                self.root.after(0, lambda t=total: self._auto_download_images_from_script_ui(t))
        except Exception as e:
            self.stream_queue.put(("error", str(e)))

    # ---------------- Queue & Save ----------------
    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.stream_queue.get_nowait()
                if kind == "text":
                    self.output_text.insert(tk.END, payload)
                    self.output_text.see(tk.END)
                elif kind == "done":
                    secs = payload
                    self.status_var.set(f"Done in {secs:.1f}s.")
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                elif kind == "error":
                    messagebox.showerror(APP_TITLE, f"Σφάλμα: {payload}")
                    self.status_var.set("Error.")
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                elif kind == "tts_done":
                    self.status_var.set(str(payload))
                    try:
                        messagebox.showinfo(APP_TITLE, str(payload))
                    except Exception:
                        pass
                elif kind == "tts_error":
                    self.status_var.set(str(payload))
                    try:
                        messagebox.showerror(APP_TITLE, str(payload))
                    except Exception:
                        pass
                self.stream_queue.task_done()
        except queue.Empty:
            pass
        self.root.after(50, self._drain_queue)

    def save_output(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo(APP_TITLE, "Δεν υπάρχει περιεχόμενο για αποθήκευση.")
            return
        file = filedialog.asksaveasfilename(
            title="Αποθήκευση κειμένου",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="ai_content.txt"
        )
        if file:
            with open(file, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_var.set(f"Saved to {os.path.basename(file)}")

    # ---------------- Preview (quotes-only) ----------------
    def preview_tts(self):
        try:
            sel = self.output_text.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
        except tk.TclError:
            sel = ""
        text = sel or self.output_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo(APP_TITLE, "Δεν υπάρχει κείμενο για ανάγνωση.")
            return

        quotes = self.extract_quoted_segments(sel) if sel else self.extract_quoted_segments(text)
        if not quotes and sel:
            quotes = self.extract_quoted_segments(text)
        if not quotes:
            messagebox.showinfo(APP_TITLE, "Δεν βρέθηκαν αποσπάσματα σε διπλά εισαγωγικά για προεπισκόπηση.")
            return

        preview_list = []
        total_chars = 0
        for q in quotes:
            if total_chars and total_chars + len(q) > 400:
                break
            preview_list.append(q)
            total_chars += len(q)
            if len(preview_list) >= 2:
                break
        preview_text = "\n\n".join(preview_list)

        self.tts.enqueue({'op': 'set', 'rate': int(self.rate_var.get()), 'volume': float(self.volume_var.get())})
        self.tts.enqueue({'op': 'preview', 'text': preview_text})
        self.status_var.set("Preview (quotes-only)…")

    def stop_tts(self):
        self.tts.enqueue({'op': 'stop'})
        self.status_var.set("TTS stopped.")

    # ---------------- Helpers ----------------
    def extract_quoted_segments(self, text):
        patterns = [r'"([^"]+)"', r'“([^”]+)”', r'«([^»]+)»']
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.DOTALL):
                start = m.start()
                content = (m.group(1) or "").strip()
                if content and len(content) >= 2:
                    matches.append((start, content))
        matches.sort(key=lambda x: x[0])
        cleaned = []
        last = None
        for _, c in matches:
            if c != last:
                cleaned.append(c)
            last = c
        return cleaned

    # ---------------- Export only quotes ----------------
    def export_audio_quotes(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo(APP_TITLE, "Δεν υπάρχει περιεχόμενο για εξαγωγή.")
            return
        self._export_quotes_worker(text, ask_dir=True)

    def _export_quotes_worker(self, text, ask_dir=False):
        directory = None
        if ask_dir:
            directory = filedialog.askdirectory(title="Επιλογή φακέλου για τα audio αρχεία")
            if not directory:
                return
        else:
            directory = os.getcwd()

        prefix = self.prefix_var.get().strip() or "script_part"
        fmt = self.audio_fmt_var.get().lower()

        quotes = self.extract_quoted_segments(text)
        if not quotes:
            messagebox.showinfo(APP_TITLE, "Δεν εντοπίστηκαν αποσπάσματα σε διπλές αποστρόφους.")
            return

        self.tts.enqueue({'op': 'set', 'rate': int(self.rate_var.get()), 'volume': float(self.volume_var.get())})

        if fmt == "mp3":
            tasks = []
            for i, seg in enumerate(quotes, start=1):
                mp3_path = os.path.join(directory, f"{prefix}_{i:02d}.mp3")
                tasks.append((seg, mp3_path))
            self.status_var.set(f"Exporting {len(tasks)} MP3 file(s) (via ffmpeg)…")
            self.tts.enqueue({'op': 'export_and_convert_mp3', 'tasks': tasks})
        else:
            if platform.system().lower().startswith("win") and fmt == "aiff":
                fmt = "wav"
            tasks = []
            for i, seg in enumerate(quotes, start=1):
                path = os.path.join(directory, f"{prefix}_{i:02d}.{fmt}")
                tasks.append((seg, path))
            self.status_var.set(f"Exporting {len(tasks)} {fmt.upper()} file(s)…")
            self.tts.enqueue({'op': 'export', 'tasks': tasks})

    # ---------------- STOCK IMAGES ----------------
    def _open_keys_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("API Keys")
        win.geometry("560x200")
        ttk.Label(win, text="Pexels API Key (Authorization header):").pack(anchor="w", padx=10, pady=(10, 2))
        pexels_var = tk.StringVar(value=self.api_keys.get("pexels", ""))
        ttk.Entry(win, textvariable=pexels_var, width=70).pack(anchor="w", padx=10)

        ttk.Label(win, text="Pixabay API Key (key=... param):").pack(anchor="w", padx=10, pady=(10, 2))
        pixabay_var = tk.StringVar(value=self.api_keys.get("pixabay", ""))
        ttk.Entry(win, textvariable=pixabay_var, width=70).pack(anchor="w", padx=10)

        ttk.Label(win, text="Unsplash Access Key (Authorization: Client-ID ...):").pack(anchor="w", padx=10, pady=(10, 2))
        unsplash_var = tk.StringVar(value=self.api_keys.get("unsplash", ""))
        ttk.Entry(win, textvariable=unsplash_var, width=70).pack(anchor="w", padx=10)

        def save_keys():
            self.api_keys["pexels"] = pexels_var.get().strip()
            self.api_keys["pixabay"] = pixabay_var.get().strip()
            self.api_keys["unsplash"] = unsplash_var.get().strip()
            try:
                with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.api_keys, f, ensure_ascii=False, indent=2)
                messagebox.showinfo(APP_TITLE, "API keys saved.")
            except Exception as e:
                messagebox.showerror(APP_TITLE, f"Failed to save keys: {e}")
            win.destroy()

        ttk.Button(win, text="Save", command=save_keys).pack(pady=12)

    def load_api_keys(self):
        data = None
        try:
            with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            try:
                here = _Path(__file__).parent / API_KEYS_FILE
                if here.exists():
                    data = json.loads(here.read_text(encoding='utf-8'))
            except Exception:
                data = None
        if not data:
            data = {
                "pexels": os.environ.get("PEXELS_API_KEY", ""),
                "pixabay": os.environ.get("PIXABAY_API_KEY", ""),
                "unsplash": os.environ.get("UNSPLASH_ACCESS_KEY", ""),
            }
        self.api_keys.update({k: (v or "") for k, v in data.items()})

    # ---- Manual download (query or selected quote) ----
    def _download_images(self):
        prov = self.provider_var.get()
        use_sel = self.use_quote_var.get()
        query = self.stock_query_var.get().strip()

        if use_sel:
            try:
                sel = self.output_text.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
            except tk.TclError:
                sel = ""
            quotes = self.extract_quoted_segments(sel) if sel else self.extract_quoted_segments(self.output_text.get("1.0", tk.END))
            if quotes:
                query = quotes[0][:100]
        if not query:
            messagebox.showwarning(APP_TITLE, "Δώσε όρο αναζήτησης ή επίλεξε «Use selected quote as query».")
            return

        max_items = max(1, int(self.max_items_var.get()))

        # Keys
        if prov == "Pexels" and not self.api_keys.get("pexels"):
            messagebox.showwarning(APP_TITLE, "Βάλε Pexels API key (API Keys…).")
            return
        if prov == "Pixabay" and not self.api_keys.get("pixabay"):
            messagebox.showwarning(APP_TITLE, "Βάλε Pixabay API key (API Keys…).")
            return
        if prov.startswith("Unsplash") and not self.api_keys.get("unsplash"):
            messagebox.showwarning(APP_TITLE, "Βάλε Unsplash Access Key (API Keys…).")
            return

        dest = filedialog.askdirectory(title="Φάκελος αποθήκευσης (εικόνες)")
        if not dest:
            return

        try:
            urls = []
            if prov == "Pixabay":
                urls = self._pixabay_search_images(query, max_items)
            elif prov == "Pexels":
                urls = self._pexels_search_images(query, max_items)
            elif prov.startswith("Unsplash"):
                urls = self._unsplash_search_images(query, max_items)
            else:
                urls = self._google_images_scrape(query, max_items)

            if not urls:
                messagebox.showinfo(APP_TITLE, "Δεν βρέθηκαν αποτελέσματα.")
                return

            saved = 0
            for i, url in enumerate(urls, start=1):
                name = self._make_image_filename(query, i, url)
                path = os.path.join(dest, name)
                if self._download_image(url, path):
                    saved += 1

            self.status_var.set(f"Αποθηκεύτηκαν {saved}/{len(urls)} εικόνες στο {dest}")
            messagebox.showinfo(APP_TITLE, f"Ολοκληρώθηκε: {saved} εικόνες.")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Σφάλμα λήψης: {e}")

    # ---- MAIN-THREAD UI HELPERS ----
    def _ask_directory_main(self, title):
        return filedialog.askdirectory(title=title) or ""

    def _auto_download_images_from_prompt_ui(self, prompt):
        dest = self._ask_directory_main("Φάκελος για αυτόματο κατέβασμα εικόνων (topic)")
        if not dest:
            return
        threading.Thread(target=self._auto_download_images_from_prompt_worker, args=(prompt, dest), daemon=True).start()

    def _auto_download_images_from_script_ui(self, text):
        dest = self._ask_directory_main("Βάλε φάκελο για εικόνες ανά quote")
        if not dest:
            return
        threading.Thread(target=self._auto_download_images_from_script_worker, args=(text, dest), daemon=True).start()

    # ---- Auto-download by topic (worker) ----
    def _extract_topic_from_prompt(self, prompt: str) -> str:
        p = (prompt or "").strip().strip('"').strip("'")
        low = p.lower()
        if " about " in low:
            pos = low.rfind(" about ") + len(" about ")
            return p[pos:].strip(' "\'.,:;')
        return p

    def _auto_download_images_from_prompt_worker(self, prompt: str, dest: str):
        try:
            query = self._extract_topic_from_prompt(prompt)
            if not query:
                return
            max_items = max(1, int(self.max_items_var.get()))
            prov = self.provider_var.get()

            # Keys
            if prov == "Pexels" and not self.api_keys.get("pexels"):
                self.stream_queue.put(("tts_error", "Pexels: λείπει API key."))
                return
            if prov == "Pixabay" and not self.api_keys.get("pixabay"):
                self.stream_queue.put(("tts_error", "Pixabay: λείπει API key."))
                return
            if prov.startswith("Unsplash") and not self.api_keys.get("unsplash"):
                self.stream_queue.put(("tts_error", "Unsplash: λείπει Access Key."))
                return

            # URLs
            if prov == "Pixabay":
                urls = self._pixabay_search_images(query, max_items)
            elif prov == "Pexels":
                urls = self._pexels_search_images(query, max_items)
            elif prov.startswith("Unsplash"):
                urls = self._unsplash_search_images(query, max_items)
            else:
                urls = self._google_images_scrape(query, max_items)

            if not urls:
                self.stream_queue.put(("tts_error", f"Δεν βρέθηκαν εικόνες για: {query}"))
                return

            saved = 0
            for i, url in enumerate(urls, start=1):
                name = self._make_image_filename(query, i, url)
                path = os.path.join(dest, name)
                if self._download_image(url, path):
                    saved += 1

            self.stream_queue.put(("tts_done", f"Auto-downloaded {saved}/{len(urls)} εικόνες για: {query}"))
        except Exception as e:
            self.stream_queue.put(("tts_error", f"Σφάλμα auto-download: {e}"))

    # ---- Auto-download per quote (worker) ----
    def _auto_download_images_from_script_worker(self, text: str, dest: str):
        quotes = self.extract_quoted_segments(text or "")
        if not quotes:
            self.stream_queue.put(("tts_error", "Δεν βρέθηκαν quoted αποσπάσματα για κατέβασμα εικόνων."))
            return

        prov = self.provider_var.get()
        k = max(1, int(self.per_quote_var.get()))
        make_sub = bool(self.subfolders_var.get())

        if prov == "Pexels" and not self.api_keys.get("pexels"):
            self.stream_queue.put(("tts_error", "Pexels: λείπει API key."))
            return
        if prov == "Pixabay" and not self.api_keys.get("pixabay"):
            self.stream_queue.put(("tts_error", "Pixabay: λείπει API key."))
            return
        if prov.startswith("Unsplash") and not self.api_keys.get("unsplash"):
            self.stream_queue.put(("tts_error", "Unsplash: λείπει Access Key."))
            return

        saved_total = 0
        uniq = set()
        for i, q in enumerate(quotes, start=1):
            query = q[:120]
            try:
                if prov == "Pixabay":
                    urls = self._pixabay_search_images(query, k * 2)
                elif prov == "Pexels":
                    urls = self._pexels_search_images(query, k * 2)
                elif prov.startswith("Unsplash"):
                    urls = self._unsplash_search_images(query, k * 2)
                else:
                    urls = self._google_images_scrape(query, k * 3)
            except Exception as e:
                self.stream_queue.put(("tts_error", f"Σφάλμα αναζήτησης για quote {i}: {e}"))
                continue

            if not urls:
                continue

            base_dir = dest
            if make_sub:
                base_dir = os.path.join(dest, f"scene_{i:02d}")
                os.makedirs(base_dir, exist_ok=True)

            taken = 0
            for url in urls:
                if url in uniq:
                    continue
                uniq.add(url)
                taken += 1
                name = self._make_image_filename(f"scene_{i:02d}", taken, url)
                path = os.path.join(base_dir, name)
                if self._download_image(url, path):
                    saved_total += 1
                if taken >= k:
                    break

        if saved_total == 0:
            self.stream_queue.put(("tts_error", "Δεν αποθηκεύτηκε καμία εικόνα."))
        else:
            self.stream_queue.put(("tts_done", f"Κατέβασαν {saved_total} εικόνες συνολικά για {len(quotes)} quotes."))

    # ---- Download primitives ----
    def _make_image_filename(self, query, idx, url):
        base = re.sub(r'[^A-Za-z0-9\-_. ]+', '', query).strip().replace(' ', '_') or "asset"
        ext = os.path.splitext(url.split('?')[0])[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            ext = '.jpg'
        return f"img_{base}_{idx:02d}{ext}"

    def _download_image(self, url, path):
        """
        Κατεβάζει το URL, επιβεβαιώνει ότι είναι image/*, επαληθεύει με Pillow (αν υπάρχει)
        και αν είναι ενεργό το 'Convert to JPG', σώζει ως JPG για συμβατότητα.
        """
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
            "Referer": "https://www.google.com/"
        }
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                ct = (r.headers.get('Content-Type') or '').lower()
                if not ct.startswith('image/'):
                    self.status_var.set(f"Skipped non-image content: {url}")
                    return False

                tmp_dir = tempfile.mkdtemp(prefix="img_dl_")
                tmp_path = os.path.join(tmp_dir, "tmp_img.bin")
                with open(tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Optional: Pillow validate/convert
            try:
                from PIL import Image, UnidentifiedImageError
                try:
                    im = Image.open(tmp_path)
                    im.load()
                except UnidentifiedImageError:
                    self.status_var.set(f"Invalid image data (skip): {url}")
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return False

                if self.convert_jpg_var.get():
                    base, _ = os.path.splitext(path)
                    out_path = base + ".jpg"
                    if im.mode not in ("RGB",):
                        try:
                            im = im.convert("RGB")
                        except Exception:
                            pass
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    im.save(out_path, "JPEG", quality=92, optimize=True)
                else:
                    fmt = (getattr(im, "format", None) or "").lower()
                    base, ext = os.path.splitext(path)
                    if fmt and ext.lower() != f".{fmt}":
                        path = base + f".{fmt}"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    shutil.copyfile(tmp_path, path)
            except Exception:
                # Χωρίς Pillow: κράτα bytes, ή διόρθωσε σε .jpg αν ct=jpeg
                base, ext = os.path.splitext(path)
                if 'image/jpeg' in ct and ext.lower() != '.jpg':
                    path = base + '.jpg'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                shutil.move(tmp_path, path)
            finally:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

            return True
        except Exception as e:
            self.status_var.set(f"Download failed for {url}: {e}")
            return False

    # -------- Provider calls --------
    def _pixabay_search_images(self, query, max_items):
        key = self.api_keys.get("pixabay", "")
        params = {
            "key": key,
            "q": query,
            "image_type": "photo",
            "per_page": min(200, max_items),
            "safesearch": "true",
        }
        url = "https://pixabay.com/api/?" + urlencode(params)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        out = []
        for h in data.get("hits", []):
            link = h.get("largeImageURL") or h.get("webformatURL")
            if link:
                out.append(link)
            if len(out) >= max_items:
                break
        return out

    def _pexels_search_images(self, query, max_items):
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.api_keys.get("pexels", "")}
        params = {"query": query, "per_page": min(80, max_items)}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out = []
        for p in data.get("photos", []):
            src = p.get("src", {})
            link = src.get("original") or src.get("large2x") or src.get("large") or src.get("medium")
            if link:
                out.append(link)
            if len(out) >= max_items:
                break
        return out

    def _unsplash_search_images(self, query, max_items):
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {self.api_keys.get('unsplash', '')}"}
        params = {"query": query, "per_page": min(30, max_items)}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out = []
        for res in data.get("results", []):
            urls = res.get("urls", {})
            link = urls.get("full") or urls.get("regular") or urls.get("small")
            if link:
                out.append(link)
            if len(out) >= max_items:
                break
        return out

    def _google_images_scrape(self, query, max_items):
        """
        Ελαφρύς scraper Google Images (tbm=isch) χωρίς BeautifulSoup.
        Μαζεύει http(s) links προς .jpg/.jpeg/.png/.gif/.webp και src από <img>.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.8"
        }
        from urllib.parse import quote as _q
        url = f"https://www.google.com/search?tbm=isch&hl=en&safe=active&q={_q(query)}"
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        html = r.text

        urls, seen = [], set()

        # 1) img src="http..."
        for m in re.finditer(r'<img[^>]+(?:src|data-src)=["\'](https?://[^"\']+)["\']', html, re.IGNORECASE):
            link = m.group(1)
            if link not in seen:
                seen.add(link)
                urls.append(link)
                if len(urls) >= max_items:
                    return urls

        # 2) ωμές διευθύνσεις αρχείων
        for m in re.finditer(r'https?://[^\s"\']+\.(?:jpg|jpeg|png|gif|webp)', html, re.IGNORECASE):
            link = m.group(0)
            if link not in seen:
                seen.add(link)
                urls.append(link)
                if len(urls) >= max_items:
                    break
        return urls[:max_items]


if __name__ == "__main__":
    root = tk.Tk()
    try:
        from tkinter import ttk as _ttk
        style = _ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = AiContentApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.tts.shutdown(), root.destroy()))
    root.mainloop()
