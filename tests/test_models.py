import struct
import unittest

from kevin.models import MAX_FILE, normalize_url, parse_header, validate


def header_fixture():
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    metadata = [("general.architecture", 8, string("llama")),
                ("llama.block_count", 4, struct.pack("<I", 30)),
                ("llama.embedding_length", 4, struct.pack("<I", 576)),
                ("llama.attention.head_count", 4, struct.pack("<I", 9)),
                ("llama.attention.head_count_kv", 4, struct.pack("<I", 3))]
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, kind, value in metadata:
        data += string(key) + struct.pack("<I", kind) + value
    return data + string("token_embd.weight") + struct.pack("<IQQIQ", 2, 576, 49152, 8, 0)


class ModelValidation(unittest.TestCase):
    def setUp(self):
        self.header = parse_header(header_fixture())

    def test_file_urls_are_normalized(self):
        url = "https://huggingface.co/owner/repo/blob/main/sub/model.gguf?download=true"
        self.assertEqual(normalize_url(url), "https://huggingface.co/owner/repo/resolve/main/sub/model.gguf")

    def test_rejects_non_hf_urls_and_non_models(self):
        for url in ["http://huggingface.co/a/b/resolve/main/m.gguf", "https://evil.com/a/b/resolve/main/m.gguf",
                    "https://huggingface.co/a/b", "https://huggingface.co/a/b/blob/main/m.safetensors",
                    "https://huggingface.co/a/b/resolve/main/m-00001-of-00002.gguf"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                normalize_url(url)

    def test_bounded_parser_and_architecture(self):
        with self.assertRaises(EOFError):
            parse_header(header_fixture()[:-3])
        with self.assertRaises(ValueError):
            parse_header(b"HTML" + header_fixture()[4:])
        self.header["metadata"]["general.architecture"] = "qwen3next"
        with self.assertRaisesRegex(ValueError, "Architecture"):
            validate(self.header, 150_000_000, memory=8*1024**3)

    def test_memory_and_size_gates(self):
        info = validate(self.header, 150_000_000, memory=8*1024**3)
        self.assertEqual(info["kv_heads"], 3)
        self.assertGreater(info["estimated_memory"], info["size"])
        with self.assertRaisesRegex(ValueError, "RAM"):
            validate(self.header, 150_000_000, memory=100_000_000)
        with self.assertRaisesRegex(ValueError, "1 GiB"):
            validate(self.header, MAX_FILE+1, memory=8*1024**3)

    def test_rejects_unsupported_cache_layouts(self):
        for key, value in [("llama.attention.head_count_kv", 4), ("llama.expert_count", 8),
                           ("llama.attention.sliding_window", 64), ("llama.attention.key_length", 32), ("split.count", 2)]:
            with self.subTest(key=key):
                h = parse_header(header_fixture()); h["metadata"][key] = value
                with self.assertRaises(ValueError):
                    validate(h, 150_000_000, memory=8*1024**3)

    def test_rejects_unsupported_quantization(self):
        self.header["tensor_types"] = [999]
        with self.assertRaisesRegex(ValueError, "quantization"):
            validate(self.header, 150_000_000, memory=8*1024**3)


if __name__ == "__main__":
    unittest.main()
