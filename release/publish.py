"""Publish a release bundle to Hugging Face: create the model repo private (if new), upload the bundle and its
card, download it again and check every file's SHA-256 against the local bundle. With --public, make the repo
public after the check passes. Needs a write token (hf auth login).
Usage: python release/publish.py BUNDLE_DIR CARD_MD REPO_ID [--public]"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(bundle, card, repo, *flags):
    bundle, api = Path(bundle), HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "bundle"
        shutil.copytree(bundle, stage)
        shutil.copy(card, stage / "README.md")
        api.upload_folder(repo_id=repo, folder_path=str(stage), commit_message=f"Release {bundle.name}")
        local = {p.name: sha(p) for p in stage.iterdir()}
    got = Path(snapshot_download(repo, local_dir=tempfile.mkdtemp()))
    remote = {p.name: sha(p) for p in got.iterdir() if p.is_file() and not p.name.startswith(".")}
    bad = sorted(f for f in local if remote.get(f) != local[f])
    print(f"{repo}: {len(local)} files uploaded, {len(local) - len(bad)} verified by download")
    if bad:
        raise SystemExit(f"MISMATCH: {bad}")
    if "--public" in flags:
        api.update_repo_settings(repo, private=False)
        print(f"{repo} is public: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main(*sys.argv[1:])
