# Third-party components

- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT license. Downloaded separately by setup into `.deps/llama.cpp`; its license is included in that checkout. Kavin builds against its unmodified public API and ggml scheduler callback.
- [nlohmann/json](https://github.com/nlohmann/json), MIT license. Used from llama.cpp's vendored distribution, including its license notices.
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), Apache 2.0. A user-selected model is downloaded separately. The recommended GGUF is distributed by [ggml-org](https://huggingface.co/ggml-org/SmolLM2-135M-GGUF).
- Interface structure, alignment and color language are adapted from the user's littleLLAMA project. Kavin does not import or alter its model, training pipeline, or inference runtime.

Model publishers may use different licenses. Their original licenses and notices continue to apply to downloaded models.
# Recorded web demo

The demo data was captured by KeVin from the real
[SmolLM2 135M Q8_0 model](https://huggingface.co/ggml-org/SmolLM2-135M-GGUF),
derived from [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
The repository contains recorded intermediate tensors and retained projection
weight slices for visualization, not the model GGUF. Capture metadata and the
model source are stored in `demo/data/manifest.json`.

