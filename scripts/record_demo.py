#!/usr/bin/env python3
"""Capture a bounded, real SmolLM inference session for the static web demo."""
import argparse
import gzip
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def record(model, output, prompt="Once upon a time", limit=5):
    if limit < 1 or limit > 20:
        raise ValueError("Demo token limit must be between 1 and 20")
    output.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([str(ROOT / "build/bin/kavin-engine")], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    def call(command):
        process.stdin.write(json.dumps(command) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Inference engine exited during demo recording")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Inference failed"))
        return result
    try:
        loaded = call({"op": "load", "path": str(model.resolve())})
        trace = call({"op": "prefill", "prompt": prompt})
        frames = []
        def frame(value):
            return {k:v for k,v in value.items() if k not in
                    ("steps", "weights", "model", "layer", "head", "kv_head")}
        frames.append(frame(trace))
        for _ in range(limit):
            trace = call({"op": "step"})
            frames.append(frame(trace))
            if trace["ended"] or trace["full"]:
                break
        model_info = loaded["model"]
        for layer in range(model_info["layers"]):
            for head in range(model_info["heads"]):
                view = call({"op": "view", "layer": layer, "head": head})
                raw = json.dumps(view, separators=(",", ":"), allow_nan=False).encode()
                (output / f"layer-{layer}-head-{head}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
            print(f"Captured layer {layer+1}/{model_info['layers']}", flush=True)
        manifest = {"schema": "kevin-recorded-demo-v1", "model": model_info, "prompt": prompt,
                    "generationLimit": len(frames)-1, "frames": frames,
                    "capturedAt": datetime.now(timezone.utc).isoformat(),
                    "source": {"file": model.name, "url": "https://huggingface.co/ggml-org/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M-Q8_0.gguf"}}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {model_info['layers'] * model_info['heads']} real layer/head views; {len(frames)-1} generated tokens", flush=True)
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        process.stdout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/data")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    record(args.model, args.output, args.prompt, args.limit)
