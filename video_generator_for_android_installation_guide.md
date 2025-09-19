# Auto Movie Maker 🎬

Turn a single **Title** into a complete video:
- Generates narration text with **Pollinations AI**
- Converts narration to speech via **gTTS** (online) or **espeak-ng** (offline fallback)
- Fetches images using your `portal_test.py`
- Composes a slideshow video with **MoviePy**
- Simple **Tkinter** GUI (works on Android using **Termux:X11**)

> **Tested target**: Android device, Termux (F-Droid/GitHub build), Termux:X11, Python 3.12+  
> **Do NOT** use the deprecated Play-Store Termux — it’s missing packages you’ll need.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Install Termux & Termux:X11](#install-termux--termuxx11)
3. [First-time Termux setup](#first-time-termux-setup)
4. [Install system packages](#install-system-packages)
5. [Install Python packages](#install-python-packages)
6. [Verify each component (sanity checks)](#verify-each-component-sanity-checks)
7. [Project layout & files](#project-layout--files)
8. [Run the app](#run-the-app)
9. [Troubleshooting](#troubleshooting)
10. [FAQ & Tips](#faq--tips)
11. [License](#license)

---

## Prerequisites

- Android device (64-bit recommended).
- Internet access (required for **gTTS** and **Pollinations**; offline TTS uses **espeak-ng**).
- Enough free storage for images and rendered videos (the app saves under `/sdcard/Download/...`).

---

## Install Termux & Termux:X11

1) **Install Termux** (not from Play Store):
- From **F-Droid**: https://f-droid.org/en/packages/com.termux/  
- or **GitHub Releases**: https://github.com/termux/termux-app/releases

2) **Install Termux:X11**:
- Download APK from: https://github.com/termux/termux-x11/releases  
- Install `termux-x11-universal-debug.apk` (or arm64 build if you know your ABI).

> Enable “Install from unknown sources” on your device if prompted.

---

## First-time Termux setup

Open **Termux** and run:

```bash
# Update & upgrade
pkg update -y && pkg upgrade -y

# Storage permission (to access /sdcard/Download)
termux-setup-storage
Now install the X11 repo and the X11 helper:

bash
Copy code
pkg install -y x11-repo
pkg install -y termux-x11-nightly
Start the Termux:X11 app once (it shows a black/gray X server screen).
Back in Termux, set the display for GUI apps:

bash
Copy code
export DISPLAY=:0
# Optional: make it persistent for every new shell
echo 'export DISPLAY=:0' >> ~/.bashrc
Install system packages
We’ll install Python, Tkinter, ffmpeg (with ffplay), offline TTS, and image libs for Pillow:

bash
Copy code
pkg install -y python
pkg install -y python-tkinter tk          # Tkinter GUI support
pkg install -y ffmpeg                     # brings ffplay for audio playback
pkg install -y espeak-ng                  # offline TTS fallback
pkg install -y python-numpy               # MoviePy depends on NumPy (use Termux package!)

# (Recommended for Pillow build support)
pkg install -y libjpeg-turbo libpng freetype littlecms libtiff openjpeg zlib

# (Optional players / tools)
pkg install -y mpv termux-api xorg-xclock
Why python-numpy (not pip install numpy)?
Pip wheels are not built for Android (bionic). Termux’s python-numpy is precompiled and fast.

Install Python packages
bash
Copy code
python -m pip install --upgrade pip setuptools wheel
python -m pip install moviepy imageio pillow gTTS requests
MoviePy v2 changed some APIs (set_* → with_*).
The provided script supports both v2 (preferred) and v1 (fallback).

Verify each component (sanity checks)
A. Tkinter + X11

bash
Copy code
# Make sure Termux:X11 app is open, then:
python - <<'PY'
import os; os.environ.setdefault("DISPLAY", ":0")
import tkinter as tk
root = tk.Tk()
root.title("Tk OK")
tk.Label(root, text="Tkinter + X11 is working!").pack(padx=20, pady=20)
root.after(1500, root.destroy)
root.mainloop()
PY
If you see a small window in the Termux:X11 screen, Tkinter works.

B. ffmpeg / ffplay

bash
Copy code
ffmpeg -version
which ffplay
C. espeak-ng (offline TTS)

bash
Copy code
espeak-ng -v el "Δοκιμή ελληνικού λόγου εκτός σύνδεσης"
D. Python libs

bash
Copy code
python - <<'PY'
import numpy, moviepy, PIL, requests
print("NumPy:", numpy.__version__)
print("MoviePy:", getattr(moviepy,'__version__','?'))
print("Pillow:", PIL.__version__)
print("Requests OK")
PY
Project layout & files
Place these files in the same directory (e.g., /sdcard/Download/autmoviemaker/):

graphql
Copy code
autmoviemaker/
├── autmoviemaker.py    # the Tkinter GUI app
└── portal_test.py      # must provide: dallemini_generate(), cover_resize_pil()
portal_test.py is your module that knows how to fetch images (e.g., DALL-E mini style) and includes cover_resize_pil(img, W, H).

autmoviemaker.py imports both functions from portal_test.py.

Copy files into place (examples):

bash
Copy code
# Example: create the project folder and move files there
mkdir -p /sdcard/Download/autmoviemaker
cp /path/to/autmoviemaker.py /sdcard/Download/autmoviemaker/
cp /path/to/portal_test.py   /sdcard/Download/autmoviemaker/
On Android, working under /sdcard/Download/... is convenient and accessible by the Gallery/Files apps.

Run the app
Start Termux:X11 (keep it in foreground or split-screen).

In Termux:

bash
Copy code
cd /sdcard/Download/autmoviemaker
export DISPLAY=:0  # if you didn't add it to ~/.bashrc
python autmoviemaker.py
In the GUI:

Type a Title only (e.g., “A journey through the stars”)

Choose how many images to fetch

Pick the voice language (el, en, …)

Click Create Video (auto script)

Output path:

php-template
Copy code
/sdcard/Download/<slug>-<timestamp>/<slug>.mp4
Example: /sdcard/Download/a-journey-through-the-stars-2025xxxx-xxxx/a-journey-through-the-stars.mp4

Troubleshooting
❌ TclError: no display name and no $DISPLAY environment variable
You didn’t start the X server or didn’t export DISPLAY.

Open Termux:X11 first

In Termux: export DISPLAY=:0

Run the script again.

❌ FileNotFoundError: 'ffplay' when trying to play audio
Install ffmpeg (includes ffplay): pkg install ffmpeg
Alternative players: pkg install mpv or use Termux media player:

bash
Copy code
pkg install termux-api
termux-media-player play /sdcard/Download/hello_el.mp3
❌ Unable to locate package python-tkinter
You’re likely on the deprecated Play-Store Termux build.
Install Termux from F-Droid/GitHub, then:

bash
Copy code
pkg update && pkg upgrade -y
pkg install x11-repo
pkg install python-tkinter tk
❌ MoviePy error: ImageSequenceClip object has no attribute 'set_audio'
You’re on MoviePy v2. Use with_audio/with_duration.
The provided script already tries v2 first and falls back to v1 methods automatically.

❌ gTTS fails (network / quota / timeout)
The script falls back to espeak-ng if installed.
Make sure pkg install espeak-ng and try again.

❌ Pillow build errors (jpeg.h, png.h, etc.)
Install image libs before pip install pillow:

bash
Copy code
pkg install libjpeg-turbo libpng freetype littlecms libtiff openjpeg zlib
Then reinstall Pillow:

bash
Copy code
python -m pip install --no-cache-dir --force-reinstall pillow
❌ requests SSL/cert issues
Usually rare on modern Termux, but if it happens: update Termux, ensure date/time are correct.

FAQ & Tips
Where are my files?
The app writes to /sdcard/Download/.... Use your phone’s Files app or Gallery to share the resulting MP4s.

How do I set DISPLAY permanently?
echo 'export DISPLAY=:0' >> ~/.bashrc (then open a new Termux session).

Can I use English narration?
Yes—change Voice lang to en. You can also tweak the system prompt inside autmoviemaker.py.

Does the script support MoviePy v1?
Yes. It tries v2’s with_audio/with_duration, then falls back to v1’s set_audio/set_duration.

Performance tips
Fewer/lower-res images render faster. Rendering on a phone can be warm/slow—plug in your device.

