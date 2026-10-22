"""Integration tests use a real pretrained GGUF, never generated weights."""
import json
import math
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("KAVIN_TEST_MODEL")


@unittest.skipUnless(MODEL and (ROOT / "build/bin/kavin-engine").exists(), "Set KAVIN_TEST_MODEL to a real GGUF after building")
class RealInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen([str(ROOT / "build/bin/kavin-engine")], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        cls.request({"op": "load", "path": str(Path(MODEL).resolve())})

    @classmethod
    def tearDownClass(cls):
        cls.process.stdin.close(); cls.process.wait(timeout=10); cls.process.stdout.close()

    @classmethod
    def request(cls, command):
        cls.process.stdin.write(json.dumps(command) + "\n"); cls.process.stdin.flush()
        result = json.loads(cls.process.stdout.readline())
        if not result.get("ok"):
            raise AssertionError(result)
        return result

    def check_attention(self, trace):
        dim = trace["model"]["head_dim"]
        for step in trace["steps"]:
            m = step["matrices"]; n = step["start"] + step["count"]
            for c in range(step["count"]):
                visible = step["start"] + c + 1
                scores = [sum(m["kt"][p][d] * m["q"][d][c] for d in range(dim)) / math.sqrt(dim) for p in range(visible)]
                maximum = max(scores); exp = [math.exp(x-maximum) for x in scores]; total = sum(exp)
                for p in range(n):
                    expected = exp[p]/total if p < visible else 0
                    # CPU F16 matmul kernels can round operands internally.
                    self.assertAlmostEqual(m["a"][p][c], expected, delta=0.001)
                captured = [m["scores"][p][c] for p in range(visible)]
                peak = max(captured)
                normalized = [math.exp(value-peak) for value in captured]
                denominator = sum(normalized)
                for p, value in enumerate(normalized):
                    self.assertAlmostEqual(m["a"][p][c], value/denominator, delta=1e-6)
                for d in range(dim):
                    expected = sum(m["vc"][d][p] * m["a"][p][c] for p in range(n))
                    self.assertAlmostEqual(m["va"][d][c], expected, delta=0.001)

    def test_actual_attention_and_cache_prefix(self):
        original = self.request({"op": "prefill", "prompt": "Once upon a time"})
        self.check_attention(original)
        for _ in range(3):
            trace = self.request({"op": "step"})
            self.assertTrue(trace["cache_prefix_verified"])
            self.check_attention(trace)
            after = trace["steps"][-1]
            for key in ("kc", "vc"):
                for r, row in enumerate(trace["steps"][0]["matrices"][key]):
                    self.assertEqual(row, after["matrices"][key][r][:len(row)])
        reference = self.request({"op": "verify"})
        self.assertTrue(reference["same_next_token"])
        self.assertLess(reference["max_logit_difference"], 0.01)

    def test_grouped_query_heads_and_layer_switch(self):
        self.request({"op": "prefill", "prompt": "The sky is"})
        for _ in range(2):
            self.request({"op": "step"})
        a = self.request({"op": "view", "layer": 0, "head": 0})
        b = self.request({"op": "view", "layer": 0, "head": 1})
        self.assertEqual(a["tokens"], b["tokens"])
        self.assertEqual(a["kv_head"], b["kv_head"])
        self.assertEqual(a["steps"][-1]["matrices"]["kc"], b["steps"][-1]["matrices"]["kc"])
        c = self.request({"op": "view", "layer": a["model"]["layers"]-1, "head": a["model"]["heads"]-1})
        self.assertEqual(len(c["steps"]), len(a["steps"]))
        self.check_attention(c)

    def test_real_projection_weights_and_runtime_products(self):
        trace = self.request({"op": "prefill", "prompt": "Once upon a time"})
        for _ in range(2):
            trace = self.request({"op": "step"})
        trace = self.request({"op": "view", "layer": 0, "head": 0})
        weights = trace["weights"]
        for key in ("wq", "wk", "wv"):
            w = weights[key]
            self.assertEqual(w["rows"], trace["model"]["head_dim"])
            self.assertEqual(w["cols"], len(trace["steps"][0]["matrices"]["x"]))
            self.assertEqual(len(w["values"]), len(w["row_indices"]))
            self.assertTrue(all(len(row) == len(w["col_indices"]) for row in w["values"]))
            self.assertLess(sum(map(len, w["values"])), w["rows"] * w["cols"])
            self.assertTrue(all(math.isfinite(v) for row in w["values"] for v in row))
        for layer, head in [(0, 0), (0, 1), (trace["model"]["layers"]-1, trace["model"]["heads"]-1)]:
            current = self.request({"op": "view", "layer": layer, "head": head})
            if layer == 0 and head == 1:
                self.assertEqual(weights["wk"], current["weights"]["wk"])
                self.assertEqual(weights["wv"], current["weights"]["wv"])
                self.assertNotEqual(weights["wq"]["values"], current["weights"]["wq"]["values"])
            report = self.request({"op": "verify_projections"})
            for kind in ("q", "k", "v"):
                self.assertLess(report[kind]["kernel_scaled_error"], 0.001, (layer, head, kind, report))

    def test_reset_starts_a_fresh_cache(self):
        self.request({"op": "prefill", "prompt": "Once upon a time"})
        self.request({"op": "step"})
        result = self.request({"op": "prefill", "prompt": "Hello"})
        self.assertEqual(len(result["steps"]), 1)
        self.assertFalse(result["cache_prefix_verified"])

    def test_context_boundary_preserves_final_cache(self):
        trace = self.request({"op": "prefill", "prompt": " a" * 127})
        self.assertEqual(len(trace["tokens"]), 127)
        trace = self.request({"op": "step"})
        self.assertTrue(trace["full"])
        self.assertEqual(len(trace["tokens"]), 128)
        self.assertTrue(trace["cache_prefix_verified"])


if __name__ == "__main__":
    unittest.main()
