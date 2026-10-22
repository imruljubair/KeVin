#!/usr/bin/env python3
"""Build the pinned llama.cpp engine without modifying upstream sources."""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVISION = "67b9b0e7f6ce45d929a4411907d3c48ec719e81c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-source", type=Path, help="Clone from an existing local llama.cpp checkout")
    args = parser.parse_args()
    for tool in ("cmake", "git"):
        if not shutil.which(tool):
            raise SystemExit(f"Install {tool} and a C++17 compiler before running setup.")
    source = ROOT / ".deps/llama.cpp"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        origin = str(args.llama_source.resolve()) if args.llama_source else "https://github.com/ggml-org/llama.cpp.git"
        subprocess.run(["git", "clone", "--filter=blob:none", origin, str(source)], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(ROOT / "build"), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(ROOT / "build"), "--target", "kavin-engine", "-j", str(min(8, os.cpu_count() or 2))], check=True)
    print("Ready. Run: python3 -m kavin.server")


if __name__ == "__main__":
    main()
