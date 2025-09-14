import os, base64, requests
from datetime import datetime

ENDPOINT = "https://bf.pcuenca.net/generate"

def dallemini_generate(prompt: str, out_dir="out", timeout=90):
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
        raise RuntimeError(f"Δεν εστάλησαν εικόνες: {data}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i, b64 in enumerate(images, 1):
        # αν τυχόν έρθει ως data:uri, κράτα μόνο το base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img = base64.b64decode(b64)
        path = os.path.join(out_dir, f"dallemini_{ts}_{i:02d}.png")
        with open(path, "wb") as f:
            f.write(img)
        paths.append(path)
    return paths

if __name__ == "__main__":
    files = dallemini_generate("a warrior")
    print("Saved:\n" + "\n".join(files))
