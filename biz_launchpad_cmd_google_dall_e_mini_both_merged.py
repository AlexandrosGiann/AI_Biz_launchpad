#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biz Launchpad CMD — Images from Google, DALL·E mini, or BOTH

Προσθήκες (vs αρχικό):
  • Νέα CLI επιλογή: --images {google|dalle|both} (default: google)
  • Υποστήριξη DALL·E mini μέσω του dallemini_scraper.dallemini_generate.
    - Αν δεν υπάρχει import, περιλαμβάνεται fallback HTTP client.
  • Λογική επιλογής εικόνων ανά SHOW: Google ή DALL·E ή συνδυασμός (Google→DALL·E fallback).

Παράδειγμα χρήσης:
  makevideo --title "AI 101" --prompt "intro to AI" --images google
  makevideo --title "AI 101" --images dalle
  makevideo --title "AI 101" --images both

Σημειώσεις:
  • Για Google, απαιτείται το google_image_scraper.py στον ίδιο φάκελο ή στο CWD.
  • Για DALL·E mini, απαιτείται σύνδεση στο internet και το πακέτο requests.
  • Προαιρετικά: moviepy ή/και opencv-python, numpy, pydub, gTTS/pyttsx3.
"""

import os
import sys
import shlex
import json
import time
import textwrap
import subprocess
import shutil
import os as _os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from enum import Enum

# ---------------- Soft imports -------------------------------------------------
def _soft_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

np = _soft_import("numpy")
PIL = _soft_import("PIL")
if not PIL:
    print("[ERROR] Pillow is required. Install: pip install pillow")
    sys.exit(1)
from PIL import Image, ImageDraw, ImageFont

_MPY = _soft_import("moviepy")
ImageClip = concatenate_videoclips = AudioFileClip = None
CompositeAudioClip = None
_MPY_V2 = False
if _MPY:
    try:
        from moviepy import ImageClip as _IC, concatenate_videoclips as _CON, AudioFileClip as _AFC
        ImageClip, concatenate_videoclips, AudioFileClip = _IC, _CON, _AFC
        _MPY_V2 = True
    except Exception:
        try:
            from moviepy.editor import ImageClip as _IC, concatenate_videoclips as _CON, AudioFileClip as _AFC
            ImageClip, concatenate_videoclips, AudioFileClip = _IC, _CON, _AFC
            _MPY_V2 = False
        except Exception:
            ImageClip = concatenate_videoclips = AudioFileClip = None
if _MPY:
    try:
        from moviepy.audio.AudioClip import CompositeAudioClip as _CAC  # v1
        CompositeAudioClip = _CAC
    except Exception:
        try:
            from moviepy.audio.compositing import CompositeAudioClip as _CAC  # v2
            CompositeAudioClip = _CAC
        except Exception:
            CompositeAudioClip = None

pyttsx3 = _soft_import("pyttsx3")
edge_tts = _soft_import("edge_tts")
gtts = _soft_import("gtts")
pydub = _soft_import("pydub")
cv2 = _soft_import("cv2")
ollama_mod = _soft_import("ollama")

# ---------------- DALL·E mini integration ------------------------------------
dallemini_scraper = _soft_import("dallemini_scraper")

def _dallemini_fallback_generate(prompt: str, out_dir: str = "out", timeout: int = 90):
    """Fallback HTTP client για DALL·E mini, αν δεν υπάρχει το module."""
    import base64, requests
    from datetime import datetime
    ENDPOINT = os.environ.get("DALLEMINI_ENDPOINT", "https://bf.pcuenca.net/generate")
    os.makedirs(out_dir, exist_ok=True)
    r = requests.post(
        ENDPOINT,
        json={"prompt": prompt},
        headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    images = data.get("images", [])
    if not images:
        raise RuntimeError(f"No images returned: {data}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i, b64 in enumerate(images, 1):
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img = base64.b64decode(b64)
        path = os.path.join(out_dir, f"dallemini_{ts}_{i:02d}.png")
        with open(path, "wb") as f:
            f.write(img)
        paths.append(path)
    return paths

if dallemini_scraper and hasattr(dallemini_scraper, "dallemini_generate"):
    dallemini_generate = dallemini_scraper.dallemini_generate  # type: ignore[attr-defined]
else:
    dallemini_generate = _dallemini_fallback_generate

# ---------------- Utils --------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in s.strip()).strip("-_")[:64] or "video"

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def get_audio_duration(path: Path) -> float:
    if pydub is not None:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(str(path))
            return float(len(seg)/1000.0)
        except Exception:
            pass
    try:
        if str(path).lower().endswith(".wav"):
            import wave, contextlib
            with contextlib.closing(wave.open(str(path), "rb")) as wf:
                return wf.getnframes()/float(wf.getframerate())
    except Exception:
        pass
    if has_ffmpeg():
        try:
            out = subprocess.check_output(
                ["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(path)]
            )
            import json as _json
            data = _json.loads(out.decode("utf-8","ignore"))
            d = float(data.get("format",{}).get("duration",0.0) or 0.0)
            if d > 0: return d
        except Exception:
            pass
    return 0.0

def seconds_for_words(words: int, wpm: int=150) -> float:
    return max(1.5, (words/max(1,wpm))*60.0)

# --- MoviePy helpers -----------------------------------------------------------
def clip_with_duration(clip, dur):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(dur)
    if hasattr(clip, "set_duration"):
        return clip.set_duration(dur)
    try:
        clip.duration = dur
    except Exception:
        pass
    return clip

def clip_with_audio(clip, audio):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    if hasattr(clip, "set_audio"):
        return clip.set_audio(audio)
    return clip

def safe_write_videofile(clip, path, **kw):
    if _MPY_V2:
        for bad in ("verbose", "logger", "temp_audiofile", "remove_temp"):
            kw.pop(bad, None)
    try:
        clip.write_videofile(path, **kw)
    except TypeError:
        allowed = {"fps","codec","audio_codec","preset","threads"}
        kw2 = {k:v for k,v in kw.items() if k in allowed}
        clip.write_videofile(path, **kw2)

# --- TTS (English) -------------------------------------------------------------
def tts_to_file(text: str, out_path: Path, voice_hint: Optional[str]=None, rate: float=1.0) -> Path:
    # edge-tts -> pyttsx3 -> gTTS
    if edge_tts is not None:
        try:
            import asyncio
            default_voice = "en-US-JennyNeural" if (voice_hint or "").lower().startswith("f") else "en-US-GuyNeural"
            voice = os.environ.get("EDGE_TTS_VOICE", default_voice)
            async def _run():
                await edge_tts.Communicate(text, voice=voice, rate=f"{int((rate-1)*100)}%").save(str(out_path.with_suffix(".mp3")))
            asyncio.get_event_loop().run_until_complete(_run())
            p = out_path.with_suffix(".mp3")
            if p.exists(): return p
        except Exception:
            pass
    if pyttsx3 is not None:
        try:
            eng = pyttsx3.init()
            base = eng.getProperty("rate") or 175
            eng.setProperty("rate", int(base*rate))
            eng.save_to_file(text, str(out_path.with_suffix(".wav"))); eng.runAndWait()
            p = out_path.with_suffix(".wav")
            if p.exists(): return p
        except Exception:
            pass
    if gtts is not None:
        try:
            from gtts import gTTS
            t = gTTS(text=text, lang="en")
            p = out_path.with_suffix(".mp3"); t.save(str(p))
            if p.exists(): return p
        except Exception:
            pass
    raise RuntimeError("No TTS engine succeeded. Install edge-tts or pyttsx3 or gTTS.")

# ---------------- Visuals ------------------------------------------------------
def title_card(title: str, size: Tuple[int,int]) -> Image.Image:
    W,H = size
    img = Image.new("RGB",(W,H),(8,10,20))
    d = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype("arial.ttf", int(H*0.08))
        f2 = ImageFont.truetype("arial.ttf", int(H*0.035))
    except Exception:
        f1 = ImageFont.load_default(); f2 = ImageFont.load_default()
    t = textwrap.fill(title, width=28)
    bbox = d.multiline_textbbox((0,0),t,font=f1,spacing=10); tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    d.multiline_text(((W-tw)//2, (H-th)//2 - int(H*0.06)), t, font=f1, fill=(240,240,240), spacing=10)
    sub = "Produced with Biz Launchpad"
    sb = d.textbbox((0,0), sub, font=f2); sw, sh = sb[2]-sb[0], sb[3]-sb[1]
    d.text(((W-sw)//2, (H+th)//2), sub, font=f2, fill=(180,180,200))
    return img

def scene_frame(bg_path: Optional[Path], caption: str, size: Tuple[int,int]) -> Image.Image:
    """Frame με background image + caption band (για SAY)."""
    W,H = size
    canvas = Image.new("RGB",(W,H),(15,16,24))
    d = ImageDraw.Draw(canvas)
    if bg_path and Path(bg_path).exists():
        try:
            im = Image.open(bg_path).convert("RGB")
            r = max(W/im.width, H/im.height)
            im = im.resize((int(im.width*r), int(im.height*r)))
            ox = (im.width - W)//2; oy = (im.height - H)//3
            canvas.paste(im.crop((ox, oy, ox+W, oy+H)))
            overlay = Image.new("RGBA",(W,H),(0,0,0,120))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            d = ImageDraw.Draw(canvas)
        except Exception:
            pass
    try:
        f = ImageFont.truetype("arial.ttf", int(H*0.045))
    except Exception:
        f = ImageFont.load_default()
    wrapped = textwrap.fill(caption, width=48)
    if wrapped:
        bbox = d.multiline_textbbox((0,0), wrapped, font=f, spacing=8)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        pad = 24
        box = (max(24,(W-tw)//2 - pad), H - th - int(H*0.18), min(W-24,(W+tw)//2 + pad), H - int(H*0.08))
        overlay = Image.new("RGBA",(W,H),(0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rectangle(box, fill=(0,0,0,170))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(canvas)
        d.multiline_text((box[0]+pad, box[1]+pad), wrapped, font=f, fill=(233,233,240), spacing=8)
    return canvas

def show_frame(bg_path: Optional[Path], size: Tuple[int,int]) -> Image.Image:
    """Frame μόνο με εικόνα (για SHOW)."""
    W,H = size
    canvas = Image.new("RGB",(W,H),(15,16,24))
    if bg_path and Path(bg_path).exists():
        try:
            im = Image.open(bg_path).convert("RGB")
            r = max(W/im.width, H/im.height)
            im = im.resize((int(im.width*r), int(im.height*r)))
            ox = (im.width - W)//2; oy = (im.height - H)//2
            canvas.paste(im.crop((ox, oy, ox+W, oy+H)))
            return canvas
        except Exception:
            pass
    return canvas

# ---------------- LLaMA multi-command script -----------------------------------
def llama_commands_en(title: str, prompt: Optional[str], llama_model: Optional[str]=None) -> List[dict]:
    """
    Επιστρέφει λίστα από dicts:
      {"cmd":"SAY","text":...} ή {"cmd":"SHOW","query":...}
    """
    sys.stderr.write("[makevideo] LLaMA: generating multi-command plan...\n")
    sys_prompt = (
        "You produce a step-by-step video plan as commands. "
        "Output ONLY lines, one command per line, in this exact format:\n"
        "SAY: <what the narrator says in plain English>\n"
        "SHOW: <what image/b-roll should be shown (googleable query)>\n"
        "Rules:\n"
        "- 12 to 18 total lines.\n"
        "- Start with SHOW for an establishing visual.\n"
        "- Alternate often between SHOW and SAY.\n"
        "- Be explanatory: definition, why it matters, practical examples, simple steps, recap.\n"
        "- Simple language, no fluff, ~140–220 words across all SAY lines.\n"
        "- No numbering. No extra text besides these commands."
    )
    user_prompt = f"TITLE: {title}\nTOPIC: {prompt or title}\nLANGUAGE: English only."
    model = llama_model or os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

    text = ""
    if ollama_mod is not None:
        try:
            res = ollama_mod.chat(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.7}
            )
            text = (res.get("message", {}) or {}).get("content", "").strip()
        except Exception as e:
            sys.stderr.write(f"[makevideo] LLaMA/Ollama failed ({e}). Using fallback.\n")

    if not text:
        text = (
            "SHOW: close-up of the main topic in action\n"
            "SAY: Here's what this really means in simple terms.\n"
            "SHOW: diagram explaining the core idea\n"
            "SAY: It's important because it saves time and reduces errors.\n"
            "SHOW: real-world example #1\n"
            "SAY: In practice, people use it to accomplish X.\n"
            "SHOW: real-world example #2\n"
            "SAY: Another case: it helps with Y by doing Z.\n"
            "SHOW: step-by-step checklist\n"
            "SAY: To start: define your goal, pick a tool, try a tiny version.\n"
            "SHOW: recap slide\n"
            "SAY: In short, focus on clarity and small wins. Progress beats perfection."
        )

    cmds: List[dict] = []
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if line.upper().startswith("SAY:"):
            cmds.append({"cmd":"SAY","text":line.split(":",1)[1].strip()})
        elif line.upper().startswith("SHOW:"):
            cmds.append({"cmd":"SHOW","query":line.split(":",1)[1].strip()})
    cmds = cmds[:18]
    while len(cmds) < 12:
        cmds.append({"cmd":"SHOW","query":"relevant illustrative image"})
        cmds.append({"cmd":"SAY","text":"Here's another quick example to make it stick."})
    return cmds

# ---------------- Google-only image pipeline (NON-INTERACTIVE) -----------------
def find_google_scraper_path() -> Optional[Path]:
    here = Path(__file__).parent
    for cand in [
        here / "google_image_scraper.py",
        Path.cwd() / "google_image_scraper.py",
        Path("google_image_scraper.py"),
    ]:
        if cand.exists():
            return cand
    return None

def run_scraper_noninteractive(script: Path, query: str, work_dir: Path, timeout: int = 90) -> None:
    """
    Τρέχει το google_image_scraper.py χωρίς input():
      - Αντικαθιστά builtins.input με σταθερή τιμή (query) μέσω wrapper.
    """
    ensure_dir(work_dir)
    wrapper = work_dir / "_gis_wrapper.py"
    wrapper.write_text(
        "import os, builtins, runpy\n"
        "q = os.environ.get('GIS_QUERY','')\n"
        "builtins.input = lambda prompt=None: q\n"
        f"runpy.run_path(r'{str(script)}', run_name='__main__')\n",
        encoding="utf-8"
    )
    env = dict(os.environ); env["GIS_QUERY"] = query
    try:
        subprocess.run([sys.executable, str(wrapper)], cwd=str(script.parent), env=env,
                       timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        wrapper.unlink(missing_ok=True)
    except Exception:
        pass

def _collect_new_images(base_dir: Path, since_ts: float) -> List[Path]:
    imgs = []
    for root, _, files in _os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
                p = Path(root) / f
                try:
                    if p.stat().st_mtime >= since_ts:
                        imgs.append(p)
                except Exception:
                    pass
    return sorted(imgs, key=lambda p: p.stat().st_mtime, reverse=False)

def download_one_image_google(query: str, out_dir: Path) -> Optional[Path]:
    script = find_google_scraper_path()
    if not script:
        sys.stderr.write("[makevideo] google_image_scraper.py not found; using placeholder.\n")
        return None
    ensure_dir(out_dir)
    start_ts = time.time()
    run_scraper_noninteractive(script, query, out_dir)
    for folder in [
        script.parent / f"images_{query.replace(' ','_')}",
        script.parent / "images",
        script.parent / "downloads",
        script.parent,
    ]:
        if folder.exists():
            imgs = _collect_new_images(folder, since_ts=start_ts)
            for p in imgs:
                dst = out_dir / f"g_{slugify(query)}{p.suffix.lower()}"
                try:
                    shutil.copy2(p, dst)
                    return dst
                except Exception:
                    continue
    sys.stderr.write(f"[makevideo] Google scraper produced no file for query '{query}'. Using placeholder.\n")
    return None

# ---------------- DALL·E mini helpers -----------------------------------------
def generate_one_image_dalle(query: str, out_dir: Path, timeout: int = 90) -> Optional[Path]:
    try:
        ensure_dir(out_dir)
        paths = dallemini_generate(query, out_dir=str(out_dir), timeout=timeout)
        if isinstance(paths, list) and paths:
            return Path(paths[0])
    except Exception as e:
        sys.stderr.write(f"[makevideo] DALL·E mini failed for '{query}': {e}\n")
    return None

# ---------------- Placeholders -------------------------------------------------
def _make_placeholder(out_dir: Path, text: str) -> Path:
    ensure_dir(out_dir)
    path = out_dir / f"ph_{slugify(text)}.jpg"
    W,H = (1280,720)
    img = Image.new("RGB",(W,H),(22,24,30))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        f = ImageFont.load_default()
    t = textwrap.fill(text, width=28)
    bbox = d.multiline_textbbox((0,0), t, font=f, spacing=8)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    d.multiline_text(((W-tw)//2, (H-th)//2), t, font=f, fill=(230,230,240), spacing=8, align="center")
    img.save(path, quality=92)
    return path

class ImageMode(str, Enum):
    GOOGLE = "google"
    DALLE = "dalle"
    BOTH = "both"  # πρώτα Google, μετά DALL·E ως fallback

# Κεντρική επιλογή εικόνας βάσει mode

def get_image_for_query(query: str, out_dir: Path, mode: ImageMode) -> Path:
    if mode == ImageMode.GOOGLE:
        return download_one_image_google(query, out_dir) or _make_placeholder(out_dir, query)
    elif mode == ImageMode.DALLE:
        return generate_one_image_dalle(query, out_dir) or _make_placeholder(out_dir, query)
    else:  # BOTH: δοκίμασε Google, αλλιώς DALL·E
        return (
            download_one_image_google(query, out_dir)
            or generate_one_image_dalle(query, out_dir)
            or _make_placeholder(out_dir, query)
        )

# ---------------- Video core ---------------------------------------------------
@dataclass
class VideoConfig:
    title: str
    out_path: Path
    resolution: Tuple[int,int] = (1280,720)
    fps: int = 30
    voice_rate: float = 1.0
    kenburns: bool = True
    llama_model: Optional[str] = None  # override model if provided
    image_mode: ImageMode = ImageMode.GOOGLE

@dataclass
class Segment:
    image: Path
    caption: str
    duration: float
    audio: Optional[Path] = None  # μόνο για SAY

def write_video_opencv(segments: List[Segment], size: Tuple[int,int], fps: int, kenburns: bool, out_file: Path) -> Path:
    if cv2 is None:
        raise RuntimeError("MoviePy and OpenCV missing. Install one: pip install moviepy or opencv-python.")
    W,H = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_file), fourcc, fps, (W,H))
    if not vw.isOpened():
        raise RuntimeError("Cannot open VideoWriter. Install ffmpeg or try MoviePy.")
    for seg in segments:
        try:
            frame = Image.open(seg.image).convert("RGB").resize((W,H))
        except Exception:
            frame = Image.new("RGB",(W,H),(20,20,20))
        if np is not None:
            import numpy as _np
            base = _np.array(frame)[:, :, ::-1]  # RGB->BGR
        else:
            base = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) if np else cv2.imread(str(seg.image))
            base = cv2.resize(base, (W,H))
        total = max(1, int(round(seg.duration * fps)))
        for i in range(total):
            if kenburns and np is not None:
                z = 1.0 + 0.04*(i/total)
                zw, zh = int(W*z), int(H*z)
                rz = cv2.resize(base, (zw, zh))
                x0 = (zw - W)//2; y0 = (zh - H)//2
                crop = rz[y0:y0+H, x0:x0+W]
                vw.write(crop)
            else:
                vw.write(base)
    vw.release()
    return out_file

def _ff_concat_list_content(paths: List[Path]) -> str:
    """
    Δημιουργεί περιεχόμενο για concat demuxer του ffmpeg:
      file '<escaped path>'
    Επιστρέφει string χωρίς f-strings με backslashes σε expressions.
    """
    lines = []
    for p in paths:
        s = p.as_posix()
        s_esc = s.replace("'", "'\\''")  # escape single quotes για ffmpeg list
        lines.append(f"file '{s_esc}'")
    return "\n".join(lines)

def build_timeline_audio(segments: List[Segment], work: Path) -> Optional[Path]:
    """
    Φτιάχνει audio timeline ίδιου μήκους με το video:
      - SAY -> βάζει το voice clip για τη διάρκειά του
      - SHOW/τίτλοι -> σιωπή διάρκειας segment
    """
    ensure_dir(work)
    out = work / "voiceover_timeline.m4a"

    if pydub is not None:
        try:
            from pydub import AudioSegment
            sr = 44100
            timeline = AudioSegment.silent(duration=0, frame_rate=sr)
            for seg in segments:
                if seg.audio:
                    piece = AudioSegment.from_file(str(seg.audio))
                    timeline += piece
                else:
                    timeline += AudioSegment.silent(duration=int(seg.duration*1000), frame_rate=sr)
            out = out.with_suffix(".mp3")
            timeline.export(str(out), format="mp3", bitrate="192k")
            return out
        except Exception:
            pass

    if has_ffmpeg():
        try:
            wavs = []
            for i, seg in enumerate(segments):
                w = work / f"a_{i:03d}.wav"
                if seg.audio:
                    subprocess.run(
                        ["ffmpeg","-y","-i",str(seg.audio),"-ar","44100","-ac","2",str(w)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                else:
                    subprocess.run(
                        ["ffmpeg","-y","-f","lavfi","-t",f"{seg.duration}",
                         "-i","anullsrc=cl=stereo:r=44100", str(w)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                wavs.append(w)
            lst = work / "concat_list.txt"
            lst.write_text(_ff_concat_list_content(wavs), encoding="utf-8")
            concat_wav = work / "voiceover_timeline.wav"
            subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy",str(concat_wav)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["ffmpeg","-y","-i",str(concat_wav),"-c:a","aac",str(out)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for w in wavs: w.unlink(missing_ok=True)
                lst.unlink(missing_ok=True); concat_wav.unlink(missing_ok=True)
            except Exception:
                pass
            return out
        except Exception:
            return None

    return None

class Engine:
    def __init__(self, work: Path, image_mode: ImageMode = ImageMode.GOOGLE):
        self.work = work; ensure_dir(work)
        self.image_mode = image_mode

    def build(self, cfg: VideoConfig, prompt: Optional[str]) -> Path:
        # 1) LLaMA multi-commands
        commands = llama_commands_en(cfg.title, prompt or cfg.title, cfg.llama_model)

        # 2) Εικόνες για SHOW
        imgdir = self.work / "images"; ensure_dir(imgdir)
        show_cache: dict[str, Path] = {}
        last_image: Optional[Path] = None

        # Τίτλος & Outro
        title_p = self.work/"title.jpg"; title_card(cfg.title, cfg.resolution).save(title_p, quality=95)
        outro_p = self.work/"outro.jpg"; scene_frame(None, "Thanks for watching!", cfg.resolution).save(outro_p, quality=95)

        # 3) Segments
        segments: List[Segment] = []
        segments.append(Segment(image=title_p, caption="", duration=2.0, audio=None))
        default_show_sec = 2.0

        for idx, cmd in enumerate(commands):
            if cmd.get("cmd") == "SHOW":
                q = cmd.get("query","").strip() or cfg.title
                if q not in show_cache:
                    show_cache[q] = get_image_for_query(q, imgdir, self.image_mode)
                last_image = show_cache[q]
                show_img = self.work / f"show_{idx:03d}.jpg"
                show_frame(last_image, cfg.resolution).save(show_img, quality=95)
                segments.append(Segment(image=show_img, caption="", duration=default_show_sec, audio=None))

            elif cmd.get("cmd") == "SAY":
                text = cmd.get("text","").strip()
                if not text:
                    continue
                if last_image is None:
                    last_image = get_image_for_query(cfg.title, imgdir, self.image_mode)
                say_img = self.work / f"say_{idx:03d}.jpg"
                scene_frame(last_image, text, cfg.resolution).save(say_img, quality=95)
                voice_p = tts_to_file(text, self.work / f"voice_{idx:03d}", rate=cfg.voice_rate)
                dur = get_audio_duration(voice_p) or seconds_for_words(len(text.split()))
                segments.append(Segment(image=say_img, caption=text, duration=dur, audio=voice_p))

        segments.append(Segment(image=outro_p, caption="", duration=2.0, audio=None))

        # 4) Render
        out_path = cfg.out_path; ensure_dir(out_path.parent)

        # ---- MoviePy: ήχος per-segment
        if ImageClip and concatenate_videoclips:
            clips = []
            for seg in segments:
                c = clip_with_duration(ImageClip(str(seg.image)), seg.duration)
                if seg.audio:
                    try:
                        a = AudioFileClip(str(seg.audio))
                        c = clip_with_audio(c, a)
                    except Exception:
                        pass
                clips.append(c)
            video = concatenate_videoclips(clips, method="compose")
            sys.stderr.write("[makevideo] Export via MoviePy...\n")
            safe_write_videofile(
                video,
                str(out_path),
                fps=cfg.fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=max(2, os.cpu_count() or 2),
            )
            return out_path

        # ---- OpenCV fallback: timeline audio (σιωπές σε SHOW)
        sys.stderr.write("[makevideo] MoviePy not found — OpenCV fallback.\n")
        silent = self.work / "silent.mp4"
        write_video_opencv(segments, cfg.resolution, cfg.fps, cfg.kenburns, silent)
        timeline_audio = build_timeline_audio(segments, self.work)
        if timeline_audio and has_ffmpeg():
            try:
                subprocess.run(
                    ["ffmpeg","-y","-i",str(silent),"-i",str(timeline_audio),
                     "-c:v","copy","-c:a","aac","-shortest",str(out_path)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return out_path
            except Exception:
                pass
        os.replace(silent, out_path)
        return out_path

# ---------------- CLI ----------------------------------------------------------
HELP = """
Commands:
  help
  makevideo [--title "Title"] [--prompt "LLM topic"] [--out output.mp4]
            [--res 1280x720] [--fps 30] [--rate 1.0] [--llama-model "llama3.2:latest"] [--images google|dalle|both] [--nokens]
  exit | quit

Σημειώσεις:
- Default LLaMA model: llama3.2:latest  (άλλαξέ το με --llama-model)
- Εικόνες: Google (scraper) ή DALL·E mini (API) ή BOTH (Google → DALL·E fallback).
"""

def parse_args(tokens: List[str]) -> dict:
    out = {}; it = iter(tokens)
    for t in it:
        if t.startswith("--"):
            k = t[2:]
            try:
                v = next(it)
                if v.startswith("--"):
                    out[k] = True
                    it = iter([v] + list(it))
                else:
                    out[k] = v
            except StopIteration:
                out[k] = True
    return out


def _parse_image_mode(val: Optional[str]) -> ImageMode:
    if not val:
        return ImageMode.GOOGLE
    v = (val or "").strip().lower()
    if v in ("google", "g"):
        return ImageMode.GOOGLE
    if v in ("dalle", "dalle-mini", "mini", "dm"):
        return ImageMode.DALLE
    if v in ("both", "g+dm", "all"):
        return ImageMode.BOTH
    sys.stderr.write(f"[warn] Unknown --images value '{val}', using 'google'.\n")
    return ImageMode.GOOGLE


def run_makevideo(rest: List[str]):
    args = parse_args(rest)
    title = args.get("title") or "New Video"
    res = args.get("res","1280x720"); fps = int(args.get("fps","30"))
    rate = float(args.get("rate","1.0"))
    llama_model = args.get("llama-model")
    kenburns = not ("nokens" in args)
    image_mode = _parse_image_mode(args.get("images"))
    W,H = (1280,720)
    if "x" in res.lower():
        try: W,H = map(int, res.lower().split("x"))
        except Exception: pass
    prompt = args.get("prompt") or title
    out_arg = args.get("out")
    out_dir = Path.cwd() / "biz_out"; ensure_dir(out_dir)
    work = out_dir / slugify(title); ensure_dir(work)
    out_path = Path(out_arg) if out_arg else (out_dir / f"{slugify(title)}.mp4")

    cfg = VideoConfig(
        title=title, out_path=out_path, resolution=(W,H), fps=fps,
        voice_rate=rate, kenburns=kenburns, llama_model=llama_model,
        image_mode=image_mode,
    )
    eng = Engine(work, image_mode=image_mode)
    try:
        result = eng.build(cfg, prompt)
        print(f"\n[OK] Saved video: {result}")
        print(f"[DIR] Work folder: {work}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("• Minimum: pillow. Optional: opencv-python, numpy, pydub, gTTS/pyttsx3, moviepy, requests.")
        print("• On Windows, add ffmpeg to PATH for audio muxing in fallback.")
        print("• Ensure google_image_scraper.py is next to this file or in the working directory.")


def repl():
    print("Biz Launchpad CMD — type 'help' or 'exit'")
    while True:
        try:
            line = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not line: continue
        if line.lower() in ("exit","quit"): print("Bye!"); break
        if line.lower() == "help": print(HELP); continue
        parts = shlex.split(line)
        if not parts: continue
        cmd, *rest = parts
        if cmd == "makevideo":
            run_makevideo(rest)
        else:
            print("Unknown command. Type 'help'.")

if __name__ == "__main__":
    repl()
