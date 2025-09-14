# Biz Launchpad — images from Google, DALL·E mini, or both

A lightweight CLI/REPL tool to rapidly create slideshow-style videos with voiceover. For each `SHOW` step, images can come from **Google image scraping**, **DALL·E mini** (text-to-image), or a **hybrid** flow (Google first, DALL·E mini as fallback).

## Table of contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage / Flags](#usage--flags)
- [Image modes](#image-modes)
- [How it works](#how-it-works)
- [Output structure](#output-structure)
- [Settings & Environment variables](#settings--environment-variables)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)

---

## Features
- **Three image modes** via `--images google | dalle | both`:
  - `google`: pull images using a local `google_image_scraper.py`.
  - `dalle`: generate images via DALL·E mini (text-to-image).
  - `both`: try **Google first**, then **fallback to DALL·E mini** if no image is found.
- **Scripted video plan** from an LLM: alternates `SHOW` and `SAY` lines with concise narration.
- **Voiceover**: uses `edge-tts` (Microsoft), `pyttsx3`, or `gTTS`—whichever is available.
- **Rendering** with MoviePy (preferred) or an OpenCV + ffmpeg fallback.
- **Simple interface**: built-in REPL offering a single `makevideo` command.

## Requirements
Minimum:
- Python 3.9+
- **Pillow** (`pip install pillow`)

Recommended:
- `moviepy` **or** `opencv-python` (at least one, for video export)
- `numpy` (for smoother Ken Burns effect in the OpenCV path)
- `edge-tts` or `pyttsx3` or `gTTS` (for TTS)
- `pydub` (convenient audio timeline handling)
- `ffmpeg` on your `PATH` (audio/video muxing in the fallback)
- `requests` (for DALL·E mini HTTP fallback)

For **Google** mode:
- A `google_image_scraper.py` file in the same folder or current working directory. The tool runs it **non-interactively** and copies the latest image downloaded for each query.

## Installation
```bash
# 1) Install dependencies (example)
pip install pillow moviepy opencv-python numpy pydub edge-tts gTTS pyttsx3 requests

# 2) (Optional) Add ffmpeg to PATH
# macOS (brew):   brew install ffmpeg
# Ubuntu/Debian:  sudo apt-get install ffmpeg
# Windows (choco): choco install ffmpeg
```

Make sure `google_image_scraper.py` is next to `biz_launchpad_cmd.py` (or in your working directory) if you plan to use `--images google` or `--images both`.

## Quick start
```bash
python biz_launchpad_cmd.py
# Opens the REPL with prompt: »

# Generate a video using Google images
» makevideo --title "Solar 101" --prompt "basics of solar energy" --images google

# Generate a video with DALL·E mini only
» makevideo --title "Solar 101" --images dalle

# Hybrid: Google → if not found → DALL·E mini
» makevideo --title "Solar 101" --images both
```

## Usage / Flags
REPL command: `makevideo` with the following flags:
```
makevideo [--title "Title"] [--prompt "LLM topic"] [--out output.mp4]
          [--res 1280x720] [--fps 30] [--rate 1.0]
          [--llama-model "llama3.2:latest"]
          [--images google|dalle|both] [--nokens]
```
- `--title`: video title.
- `--prompt`: topic/description for the LLM (falls back to title if omitted).
- `--out`: output mp4 path (default: `biz_out/<slug>.mp4`).
- `--res`: resolution, e.g., `1280x720`.
- `--fps`: frames per second (default: 30).
- `--rate`: TTS speaking rate (1.0 = normal).
- `--llama-model`: Ollama model name (default: `llama3.2:latest`).
- `--images`: image source (`google | dalle | both`).
- `--nokens`: disable the mild Ken Burns effect in the OpenCV fallback.

## Image modes
- **google**: Uses `google_image_scraper.py` to fetch an image matching each `SHOW` query.
- **dalle**: Uses DALL·E mini to generate an image (prompt = `SHOW` query).
- **both**: For each `SHOW`, the tool tries **Google first**. If no image is found (or an error occurs), it **falls back** to DALL·E mini. If both fail, a placeholder is created.

> Note: `both` is strictly a **fallback** flow: it does not mix or alternate sources by design.

## How it works
1. The tool builds a `SAY`/`SHOW` plan with an LLM (via `ollama`, if available). If not, it uses a simple internal fallback plan.
2. For each `SHOW`, it obtains/generates an image based on `--images` mode.
3. `SAY` scenes render with a caption card and TTS; duration matches the voice clip.
4. The video is rendered with MoviePy or, if that is unavailable, with OpenCV, followed by an ffmpeg mux step for audio.

## Output structure
By default, outputs go to `biz_out/` under a per-title subfolder (slug):
```
biz_out/
  └─ solar-101/
      ├─ images/              # downloaded/generated images per SHOW
      ├─ title.jpg            # title card
      ├─ say_XXX.jpg          # SAY cards with caption
      ├─ voice_XXX.(mp3|wav)  # TTS clips
      └─ solar-101.mp4        # final video (or at --out path)
```

## Settings & Environment variables
- `EDGE_TTS_VOICE`: voice for `edge-tts` (e.g., `en-US-JennyNeural`).
- `OLLAMA_MODEL`: default model for the plan (falls back to `llama3.2:latest`).
- `DALLEMINI_ENDPOINT`: optional endpoint for DALL·E mini (HTTP fallback). A default is set inside the script.

## Troubleshooting
- **`Pillow is required`** → `pip install pillow`.
- **No video or audio** → install `moviepy` or `opencv-python` and ensure `ffmpeg` is available.
- **`No TTS engine succeeded`** → install one of `edge-tts`, `pyttsx3`, or `gTTS`.
- **Google mode returns no images** → ensure `google_image_scraper.py` sits next to the script and runs non-interactively. The tool wraps it and passes queries automatically.
- **DALL·E mini fails** → check internet/endpoint, or set `DALLEMINI_ENDPOINT`.
- **Windows** → ensure `ffmpeg.exe` is on your PATH.

## FAQ
**Q: What exactly does `--images both` do?**  
A: For every `SHOW`, it tries Google first. If that fails, it falls back to DALL·E mini. If both fail, it generates a placeholder image.

**Q: Can I systematically mix or alternate Google and DALL·E mini?**  
A: This version only implements fallback. Alternating or split-frame layouts are easy to add—feel free to open an issue/PR.

**Q: What language is the voiceover in?**  
A: English by default (Microsoft/pyttsx3/gTTS). You can change text/voice via flags or environment variables.

**Q: Legal/ToS considerations?**  
A: If you use a scraper, ensure compliance with the sources' terms and applicable laws. Use responsibly.

## License
Add a `LICENSE` file (e.g., MIT). If you have no preference, MIT is a simple choice.

---

**Happy creating!** If you want extras (alternating Google/DALL·E per `SHOW`, split-frame, subtitles, Greek TTS, etc.), open an issue or include instructions in the README.

