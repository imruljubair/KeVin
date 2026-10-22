"""Bounded GGUF inspection and Hugging Face downloads. No model execution here."""
from __future__ import annotations

import hashlib
import os
import re
import struct
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://huggingface.co/ggml-org/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M-Q8_0.gguf"
MAX_FILE = 1024**3
MAX_HEADER = 16 * 1024**2
CONTEXT = 128
SUPPORTED_TYPES = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30}


def normalize_url(value: str) -> str:
    url = urllib.parse.urlsplit(value.strip())
    if url.scheme != "https" or url.hostname != "huggingface.co" or url.username or url.password or url.port:
        raise ValueError("Use an HTTPS Hugging Face link to a specific GGUF file.")
    parts = url.path.split("/")
    if len(parts) < 6 or parts[3] not in ("blob", "resolve") or not parts[4]:
        raise ValueError("Choose a GGUF file in Files and versions, then paste its file URL.")
    filename = urllib.parse.unquote(parts[-1])
    if any(c in filename for c in ("/", "\\", "\x00")):
        raise ValueError("Use a GGUF file URL with a plain filename.")
    if not filename.lower().endswith(".gguf"):
        raise ValueError("The URL must point to a .gguf file.")
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", filename, re.I):
        raise ValueError("Split GGUF files are not supported yet. Choose a single-file model.")
    parts[3] = "resolve"
    return urllib.parse.urlunsplit(("https", "huggingface.co", "/".join(parts), "", ""))


class HFRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")):
            raise ValueError("The download redirected outside Hugging Face hosting.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "KeVin/0.1", **(headers or {})})
    try:
        return urllib.request.build_opener(HFRedirect()).open(req, timeout=45)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise ValueError("This model requires authentication. Choose a public, ungated GGUF.") from error
        raise ValueError(f"Hugging Face returned HTTP {error.code}. Check the file URL.") from error


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise EOFError("GGUF metadata is incomplete")
        result = self.data[self.pos:self.pos+n]
        self.pos += n
        return result

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.number("Q")
        if n > MAX_HEADER:
            raise ValueError("GGUF metadata string exceeds the supported limit.")
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, kind, retain=True, depth=0):
        types = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind in types:
            value = self.number(types[kind])
            return value if retain else None
        if kind == 8:
            value = self.string()
            return value if retain else None
        if kind == 9:
            subtype, count = self.number("I"), self.number("Q")
            if depth > 1 or count > 1_000_000:
                raise ValueError("GGUF metadata array exceeds the supported limit.")
            for _ in range(count):
                self.value(subtype, False, depth + 1)
            return None
        raise ValueError(f"Unknown GGUF metadata type: {kind}")


def parse_header(data: bytes):
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise ValueError("This file is not a GGUF model.")
    version, tensors, entries = r.number("I"), r.number("Q"), r.number("Q")
    if version not in (2, 3):
        raise ValueError("Only GGUF versions 2 and 3 are supported.")
    if not 1 <= tensors <= 10000 or entries > 10000:
        raise ValueError("Unsupported GGUF metadata or tensor count.")
    meta = {}
    for _ in range(entries):
        key, kind = r.string(), r.number("I")
        value = r.value(kind)
        if value is not None:
            meta[key] = value
    tensor_types, tensor_names, parameters = set(), [], 0
    for _ in range(tensors):
        name, ndims = r.string(), r.number("I")
        if not 1 <= ndims <= 4:
            raise ValueError("Unsupported tensor dimensions.")
        count = 1
        for _ in range(ndims):
            dimension = r.number("Q")
            if not 1 <= dimension <= 1_000_000:
                raise ValueError("Unsupported tensor dimension.")
            count *= dimension
        parameters += count
        tensor_types.add(r.number("I")); r.number("Q")
        tensor_names.append(name)
    return {"metadata": meta, "tensor_types": sorted(tensor_types), "tensor_names": tensor_names, "parameters": parameters}


def available_memory():
    try:
        if os.uname().sysname == "Darwin":
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
            stats = subprocess.check_output(["vm_stat"], text=True)
            page = int(re.search(r"page size of (\d+)", stats)[1])
            pages = sum(int(re.search(rf"{key}:\s+(\d+)", stats)[1]) for key in
                        ("Pages free", "Pages inactive", "Pages speculative"))
            return min(total, pages * page)
        mem = Path("/proc/meminfo").read_text()
        return int(re.search(r"MemAvailable:\s+(\d+)", mem)[1]) * 1024
    except (OSError, ValueError, TypeError, AttributeError):
        return 2 * 1024**3


def validate(header, size, memory=None):
    m = header["metadata"]
    arch = m.get("general.architecture")
    if arch != "llama":
        raise ValueError(f"Architecture '{arch}' is not supported. Start with SmolLM2 or another standard Llama model.")
    if m.get("general.type", "model") != "model" or m.get("split.count", 1) > 1:
        raise ValueError("Adapters and split models are not supported.")
    if m.get("llama.expert_count", 0) or m.get("llama.attention.sliding_window", 0):
        raise ValueError("Mixture-of-experts and sliding-window attention are not supported yet.")
    if not header["tensor_types"] or set(header["tensor_types"]) - SUPPORTED_TYPES:
        raise ValueError("This quantization is not supported. Choose F16, BF16, Q8_0, Q4_0, or a standard K quantization.")
    required = ("block_count", "embedding_length", "attention.head_count")
    if any(not isinstance(m.get("llama." + key), int) for key in required):
        raise ValueError("Required Llama architecture metadata is missing.")
    layers, hidden, heads = [m["llama." + key] for key in required]
    kv_heads = m.get("llama.attention.head_count_kv", heads)
    if not (1 <= layers <= 48 and 16 <= hidden <= 8192 and 1 <= heads <= 64 and 1 <= kv_heads <= heads and heads % kv_heads == 0 and hidden % heads == 0):
        raise ValueError("The model's layer/head layout is outside KeVin's supported limits.")
    dim = hidden // heads
    if not 8 <= dim <= 256 or any(m.get("llama.attention." + key, dim) != dim for key in ("key_length", "value_length")):
        raise ValueError("This model uses unsupported key/value head dimensions.")
    if m.get("llama.context_length", CONTEXT) < CONTEXT:
        raise ValueError(f"The model must support at least {CONTEXT} context tokens.")
    if size <= 0 or size > MAX_FILE:
        raise ValueError("For this release, choose a single GGUF no larger than 1 GiB.")
    # CPU execution: reserve weights, working buffers, captured activations and attention history.
    trace = layers * CONTEXT * (hidden + heads * dim * 2 + kv_heads * dim * 4 + CONTEXT * heads) * 4
    estimate = int(size * 1.5 + 256 * 1024**2 + trace)
    memory = available_memory() if memory is None else memory
    if estimate > memory * 0.8:
        raise ValueError(f"Insufficient available RAM: estimated {estimate / 1024**3:.1f} GiB needed, {memory / 1024**3:.1f} GiB available.")
    return {"name": m.get("general.name", "GGUF model"), "architecture": arch, "layers": layers,
            "heads": heads, "kv_heads": kv_heads, "head_dim": dim, "context": CONTEXT,
            "parameters": header["parameters"], "size": size, "estimated_memory": estimate,
            "available_memory": memory, "warning": "CPU inference may be slow for this model." if size > 400 * 1024**2 else None}


def inspect_url(url):
    url = normalize_url(url)
    for limit in (2 * 1024**2, 8 * 1024**2, MAX_HEADER):
        with open_url(url, {"Range": f"bytes=0-{limit-1}"}) as response:
            content_range = response.headers.get("Content-Range", "")
            size = int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else int(response.headers.get("Content-Length", 0))
            if size > MAX_FILE:
                raise ValueError("For this release, choose a single GGUF no larger than 1 GiB.")
            data = response.read(limit)
        try:
            header = parse_header(data)
            info = validate(header, size)
            return {**info, "url": url, "filename": urllib.parse.unquote(urllib.parse.urlsplit(url).path.split("/")[-1])}
        except EOFError:
            continue
    raise ValueError("Model metadata exceeds the supported 16 MiB inspection limit.")


def download(info, directory, progress):
    key = hashlib.sha256(info["url"].encode()).hexdigest()[:16]
    target = directory / f"{key}-{info['filename']}"
    if target.exists() and target.stat().st_size == info["size"]:
        with target.open("rb") as file:
            validate(parse_header(file.read(MAX_HEADER)), target.stat().st_size)
        progress(1.0, "Using cached model")
        return target
    temporary = target.with_suffix(".part")
    received = 0
    try:
        with open_url(info["url"]) as response, temporary.open("wb") as output:
            while chunk := response.read(1024**2):
                received += len(chunk)
                if received > min(MAX_FILE, info["size"]):
                    raise ValueError("Downloaded file exceeds the validated size.")
                output.write(chunk)
                progress(received / info["size"], "Downloading GGUF")
        if received != info["size"]:
            raise ValueError("Incomplete download. Try again.")
        with temporary.open("rb") as file:
            validate(parse_header(file.read(MAX_HEADER)), received)
        temporary.replace(target)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
