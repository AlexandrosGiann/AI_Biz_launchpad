# autmoviemaker.py
# Βασισμένο στο δικό σου αρχείο, με προσθήκη αυτόματης παραγωγής σεναρίου
# (Pollinations generate_text), TTS, και MoviePy v2 API (with_audio/with_duration).
# Είσοδος μόνο "Title" → εικόνες + τίτλος + αφήγηση → βίντεο.

import os
import re
import shutil
import subprocess
import threading
import textwrap
import random
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox

import requests
from PIL import Image, ImageDraw, ImageFont

# Χρησιμοποιεί τις δικές σου συναρτήσεις (πρέπει να είναι στο ίδιο folder)
from portal_test import dallemini_generate, cover_resize_pil

# MoviePy (υποστήριξη v2, με fallback αν έχεις v1)
os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"
try:
    from moviepy import ImageSequenceClip, AudioFileClip  # MoviePy v2
except Exception:
    from moviepy.editor import ImageSequenceClip, AudioFileClip  # MoviePy v1 fallback

from gtts import gTTS


# ---------- TEXT GENERATION (Pollinations) ----------
def generate_text(prompt: str, system_prompt: str = "") -> str:
    """
    Ζητά μικρό αφήγημα από το text.pollinations.ai.
    Επιστρέφει καθαρό text ή, σε αποτυχία, τον ίδιο τον τίτλο (prompt).
    """
    url = "https://text.pollinations.ai"
    try:
        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "model": "openai",
            "seed": random.randint(1, 999_999_999),
            "jsonMode": False,
            "private": True,
            "stream": False,
        }
        resp = requests.post(url, json=data, stream=True, timeout=60)
        if resp.status_code == 200:
            return resp.text.strip()
        else:
            print(f"Failed to generate text. Status code: {resp.status_code}.")
    except Exception as e:
        print(f"Request failed: {e}.")
    return prompt  # fallback


# ---------- HELPERS ----------
def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s or "video"


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def make_title_card(text: str, size=(1920, 1080), bg="#0e0e0e", fg="#ffffff") -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/system/fonts/Roboto-Regular.ttf", 80)
    except Exception:
        font = ImageFont.load_default()

    # έξυπνο τύλιγμα για έως 4 γραμμές
    lines = None
    for width in range(22, 42):
        test = textwrap.wrap(text, width=width)
        if len(test) <= 4:
            lines = test
            break
    if not lines:
        lines = textwrap.wrap(text, width=40)

    # κεντράρισμα
    line_heights, max_w = [], 0
    for line in lines:
        w, h = draw.textbbox((0, 0), line, font=font)[2:]
        max_w = max(max_w, w)
        line_heights.append(h)
    total_h = sum(line_heights) + (len(lines) - 1) * 10
    y = (H - total_h) // 2
    for line in lines:
        w, h = draw.textbbox((0, 0), line, font=font)[2:]
        x = (W - w) // 2
        draw.text((x, y), line, fill=fg, font=font)
        y += h + 10
    return img


def tts_to_file(text: str, out_mp3: str, lang: str = "el") -> str:
    """
    Δοκιμάζει gTTS (online), αλλιώς espeak-ng (offline).
    Επιστρέφει path σε αρχείο ήχου (mp3 ή wav).
    """
    try:
        gTTS(text, lang=lang).save(out_mp3)
        return out_mp3
    except Exception:
        if shutil.which("espeak-ng"):
            wav = os.path.splitext(out_mp3)[0] + ".wav"
            cmd = ["espeak-ng", "-v", lang, "-s", "150", "-w", wav, text]
            subprocess.run(cmd, check=True)
            return wav
        raise


def images_to_same_size(paths):
    if not paths:
        return []
    imgs = []
    base = Image.open(paths[0]).convert("RGB")
    W, H = base.size
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = cover_resize_pil(im, W, H)
        imgs.append(im)
    return imgs


def save_temp_frames(imgs, out_dir):
    frames = []
    for i, im in enumerate(imgs):
        fp = os.path.join(out_dir, f"frame_{i:04d}.png")
        im.save(fp, format="PNG")
        frames.append(fp)
    return frames


# ---------- GUI ----------
class AutoTitleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto Title → Video")
        self.geometry("760x260")

        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Title:").grid(row=0, column=0, sticky="w")
        self.title_var = tk.StringVar(value="Το ταξίδι ενός αστεροναύτη")
        ttk.Entry(main, textvariable=self.title_var, width=64).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0)
        )

        ttk.Label(main, text="Images:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.n_images = tk.IntVar(value=6)
        ttk.Spinbox(main, from_=3, to=20, textvariable=self.n_images, width=6).grid(
            row=1, column=1, sticky="w", pady=(8, 0)
        )

        ttk.Label(main, text="Voice lang:").grid(row=1, column=2, sticky="e", pady=(8, 0))
        self.lang_var = tk.StringVar(value="el")
        ttk.Entry(main, textvariable=self.lang_var, width=6).grid(
            row=1, column=3, sticky="w", pady=(8, 0)
        )

        self.btn = ttk.Button(main, text="Create Video (auto script)", command=self.start_make)
        self.btn.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(12, 0))

        self.status = ttk.Label(main, text="Ready.", foreground="#666")
        self.status.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        for i in range(4):
            main.columnconfigure(i, weight=1)

    def set_status(self, txt: str):
        self.status.config(text=txt)
        self.update_idletasks()

    def start_make(self):
        title = self.title_var.get().strip()
        if not title:
            messagebox.showwarning("Title", "Γράψε έναν τίτλο.")
            return
        self.btn.config(state="disabled")
        threading.Thread(target=self.make_video_from_title, args=(title,), daemon=True).start()

    def make_video_from_title(self, title: str):
        try:
            slug = slugify(title)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = ensure_dir(f"/sdcard/Download/{slug}-{stamp}")
            video_out = os.path.join(out_dir, f"{slug}.mp4")
            audio_tmp = os.path.join(out_dir, "tts.mp3")

            # 1) Γεννάμε σύντομο αφήγημα
            self.set_status("Generating script text…")
            system_prompt = (
                "Γράψε συνοπτικό, φυσικό κείμενο αφήγησης για βίντεο ~15–30 δευτερολέπτων. "
                "Καθαρό κείμενο, χωρίς markdown ή λίστες."
            )
            narration = generate_text(prompt=title, system_prompt=system_prompt) or title

            # 2) Εικόνες από το δικό σου generator
            self.set_status("Generating images…")
            try:
                img_paths = dallemini_generate(title, out_dir=out_dir, timeout=120, n_images=self.n_images.get())
            except TypeError:
                img_paths = dallemini_generate(title, out_dir=out_dir, timeout=120)

            # 3) Κάρτα τίτλου (μέγεθος 1ης εικόνας ή 1920x1080)
            self.set_status("Building title card…")
            if img_paths:
                base = Image.open(img_paths[0])
                size = base.size
            else:
                size = (1920, 1080)
            title_img = make_title_card(title, size=size)
            title_path = os.path.join(out_dir, "title.png")
            title_img.save(title_path)

            # 4) Frames (title card + images), ομοιόμορφο μέγεθος
            self.set_status("Preparing frames…")
            all_imgs = [title_path] + img_paths
            pil_imgs = images_to_same_size(all_imgs)
            frame_paths = save_temp_frames(pil_imgs, out_dir)

            # 5) TTS (gTTS → espeak-ng)
            self.set_status("Generating TTS audio…")
            audio_path = tts_to_file(narration, audio_tmp, lang=self.lang_var.get())

            # 6) Video render με διάρκεια ίση με του ήχου
            self.set_status("Rendering video…")
            aclip = AudioFileClip(audio_path)
            n = max(1, len(frame_paths))
            per = max(0.3, aclip.duration / n)  # ομοιόμορφη κατανομή
            clip = ImageSequenceClip(frame_paths, durations=[per] * n)

            # MoviePy v2 API: with_audio/with_duration (fallback σε v1: set_audio/set_duration)
            try:
                clip = clip.with_audio(aclip)        # v2
                clip = clip.with_duration(aclip.duration)
            except AttributeError:
                clip = clip.set_audio(aclip)         # v1 fallback
                clip = clip.set_duration(aclip.duration)

            clip.write_videofile(
                video_out,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                threads=2,
            )

            self.set_status(f"✅ Saved: {video_out}")
        except Exception as e:
            self.set_status(f"Error: {e}")
        finally:
            self.btn.config(state="normal")


if __name__ == "__main__":
    app = AutoTitleApp()
    app.mainloop()
