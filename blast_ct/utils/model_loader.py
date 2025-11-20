import hashlib
import pathlib
import requests
import sys
import tempfile

# allow direct file downloads, not HTML viewer pages
BASE_URL = "https://huggingface.co/sakshirathi360/blast-ct/resolve/main/"

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url, dest):
    tmp = pathlib.Path(tempfile.gettempdir()) / dest.name

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                # progress
                if total > 0:
                    done = int(50 * downloaded / total)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1e6:.2f}MB/{total/1e6:.2f}MB")
                    sys.stdout.flush()

    print("\nDownload complete.")
    tmp.rename(dest)

def get_model_path(filename, expected_sha256=None):
    dest = pathlib.Path("~/.blastct/models").expanduser() / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    # already downloaded
    if dest.exists():
        if expected_sha256 is not None:
            if _sha256(dest) != expected_sha256:
                print("[blast_ct] Checksum mismatch, redownloading…")
                dest.unlink()
            else:
                return str(dest)
        else:
            return str(dest)

    print(f"[blast_ct] Downloading model: {filename}…")

    # 🔥 FIX: enforce binary download mode
    url = f"{BASE_URL}{filename}?download=1"

    _download(url, dest)

    if expected_sha256 is not None and _sha256(dest) != expected_sha256:
        dest.unlink()
        raise ValueError(f"[blast_ct] ERROR: SHA256 mismatch for {filename}")

    return str(dest)
