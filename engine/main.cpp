#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include <fstream>
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

struct Tensor {
    int64_t ne[4] = {1, 1, 1, 1};
    std::vector<float> data;
    float at(int64_t a, int64_t b = 0, int64_t c = 0, int64_t d = 0) const {
        if (a < 0 || b < 0 || c < 0 || d < 0 || a >= ne[0] || b >= ne[1] || c >= ne[2] || d >= ne[3])
            throw std::runtime_error("Unexpected tensor shape [" + std::to_string(ne[0]) + "," + std::to_string(ne[1]) + "," + std::to_string(ne[2]) + "," + std::to_string(ne[3]) + "] at [" + std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c) + "," + std::to_string(d) + "]");
        return data.at(a + ne[0] * (b + ne[1] * (c + ne[2] * d)));
    }
};

static Tensor read_tensor(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)
        throw std::runtime_error("Inspection requires floating point activation tensors");
    Tensor out;
    std::copy(t->ne, t->ne + 4, out.ne);
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    out.data.reserve(ggml_nelements(t));
    for (int64_t d = 0; d < t->ne[3]; ++d)
    for (int64_t c = 0; c < t->ne[2]; ++c)
    for (int64_t b = 0; b < t->ne[1]; ++b)
    for (int64_t a = 0; a < t->ne[0]; ++a) {
        const auto offset = a*t->nb[0] + b*t->nb[1] + c*t->nb[2] + d*t->nb[3];
        float value;
        if (t->type == GGML_TYPE_F32) std::memcpy(&value, bytes.data() + offset, 4);
        else if (t->type == GGML_TYPE_F16) {
            ggml_fp16_t half; std::memcpy(&half, bytes.data() + offset, 2);
            value = ggml_fp16_to_fp32(half);
        } else {
            uint16_t half; std::memcpy(&half, bytes.data() + offset, 2);
            uint32_t bits = uint32_t(half) << 16; std::memcpy(&value, &bits, 4);
        }
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite activation encountered");
        out.data.push_back(value);
    }
    return out;
}

struct Step {
    int start = 0, count = 0;
    double ms = 0;
    std::vector<std::map<std::string, Tensor>> layers;
};

struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int context = 128, selected_layer = 0, selected_head = 0;
    int layers = 0, heads = 0, kv_heads = 0, dim = 0;
    std::vector<llama_token> tokens;
    std::vector<Step> steps;
    std::vector<std::map<std::string, Tensor>> cache;
    Step pending;
    std::string capture_error;
    bool capturing = false, ended = false;
    gguf_context * weight_gguf = nullptr;
    mutable std::ifstream weight_file;
    mutable json weight_view;
    mutable int weight_layer = -1, weight_head = -1;
    ~Engine() { unload(); }
    void unload() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        if (weight_gguf) gguf_free(weight_gguf); weight_gguf=nullptr;weight_file.close();
        ctx = nullptr; model = nullptr; vocab = nullptr;
        tokens.clear(); steps.clear(); cache.clear(); ended = false; weight_view = nullptr; weight_layer = weight_head = -1;
    }
    static bool callback(ggml_tensor * t, bool ask, void * user) {
        auto & e = *static_cast<Engine *>(user);
        if (!e.capturing) return !ask;
        std::string name = t->name;
        const auto dash = name.rfind('-');
        if (dash == std::string::npos) return !ask;
        const auto suffix = name.substr(dash + 1);
        if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) return !ask;
        const std::string kind = name.substr(0, dash);
        if (kind != "attn_norm" && kind != "kq" && kind != "kq_soft_max" && kind != "kqv" && kind != "Qcur" && kind != "Kcur" && kind != "Vcur" && kind != "wqkv") return !ask;
        if (ask) return true;
        try {
            const int layer = std::stoi(name.substr(dash + 1));
            if (layer < 0 || layer >= e.layers) return true;
            auto & target = e.pending.layers[layer];
            if (kind == "Qcur" || kind == "Kcur" || kind == "Vcur" || kind == "wqkv") {
                if (t->op == GGML_OP_MUL_MAT) target[kind == "wqkv" ? "raw_qkv" : kind == "Qcur" ? "raw_q" : kind == "Kcur" ? "raw_k" : "raw_v"] = read_tensor(t);
                return true;
            }
            // Cache operands are read after their real F16 cache writes.
            // Capture them before a subsequent graph node can reuse the buffer.
            if (kind == "kq") {
                target["scores"] = read_tensor(t);
                target["cache_k"] = read_tensor(t->src[0]);
                target["query"] = read_tensor(t->src[1]);
            }
            else if (kind == "kqv") {
                target["cache_v"] = read_tensor(t->src[0]);
                target[kind] = read_tensor(t);
            } else target[kind] = read_tensor(t);
        } catch (const std::exception & error) { e.capture_error = error.what(); return false; }
        return true;
    }
    std::string piece(llama_token id) const {
        std::vector<char> buf(256);
        int n = llama_token_to_piece(vocab, id, buf.data(), buf.size(), 0, true);
        if (n < 0) { buf.resize(-n); n = llama_token_to_piece(vocab, id, buf.data(), buf.size(), 0, true); }
        if (n < 0) throw std::runtime_error("Token conversion failed");
        return std::string(buf.data(), n);
    }
    std::vector<llama_token> tokenize(const std::string & text) const {
        int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, false);
        if (n <= 0) throw std::runtime_error("Enter a non-empty prompt");
        std::vector<llama_token> ids(n);
        n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), n, true, false);
        if (n < 0) throw std::runtime_error("Tokenization failed");
        ids.resize(n); return ids;
    }
    std::string metadata(const char * key) const {
        char buf[1024] = {};
        llama_model_meta_val_str(model, key, buf, sizeof(buf));
        return buf;
    }
    json info() const {
        return {{"name", metadata("general.name")}, {"architecture", metadata("general.architecture")},
            {"layers", layers}, {"heads", heads}, {"kv_heads", kv_heads}, {"head_dim", dim},
            {"context", context}, {"parameters", llama_model_n_params(model)},
            {"backend", "llama.cpp · CPU inspection"}, {"cache_type", "F16"}};
    }
    void decode(const std::vector<llama_token> & ids) {
        if (tokens.size() + ids.size() > size_t(context)) throw std::runtime_error("Context limit reached. Reset with a shorter prompt.");
        pending = Step{}; pending.start = tokens.size(); pending.count = ids.size(); pending.layers.resize(layers);
        capture_error.clear(); capturing = true;
        auto batch = llama_batch_get_one(const_cast<llama_token *>(ids.data()), ids.size());
        auto begin = Clock::now();
        int status = llama_decode(ctx, batch);
        llama_synchronize(ctx); capturing = false;
        pending.ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (status != 0 || !capture_error.empty()) throw std::runtime_error("Inference failed: " + (capture_error.empty() ? std::to_string(status) : capture_error));
        for (const auto & l : pending.layers)
            for (auto key : {"query", "attn_norm", "cache_k", "cache_v", "kq_soft_max", "kqv"})
                if (!l.count(key)) throw std::runtime_error(std::string("Model runs, but inspection tensor is missing: ") + key);
        // A supported append-only run must preserve all previous cache entries.
        if (!steps.empty()) {
            for (int l = 0; l < layers; ++l) {
                const auto & old_k = cache[l].at("cache_k");
                const auto & old_v = cache[l].at("cache_v");
                const auto & new_k = pending.layers[l].at("cache_k");
                const auto & new_v = pending.layers[l].at("cache_v");
                for (int h = 0; h < kv_heads; ++h) for (size_t p = 0; p < tokens.size(); ++p) for (int d = 0; d < dim; ++d)
                    if (old_k.at(d,p,h) != new_k.at(d,p,h) || old_v.at(p,d,h) != new_v.at(p,d,h))
                        throw std::runtime_error("Cache prefix changed unexpectedly; this architecture is not supported");
            }
        }
        cache.resize(layers);
        for (int l = 0; l < layers; ++l) for (auto key : {"cache_k", "cache_v"}) {
            cache[l][key] = std::move(pending.layers[l].at(key));
            pending.layers[l].erase(key);
        }
        tokens.insert(tokens.end(), ids.begin(), ids.end());
        steps.push_back(std::move(pending));
    }
    json matrix(int rows, int cols, const std::function<float(int,int)> & get) const {
        json values = json::array();
        for (int r = 0; r < rows; ++r) {
            json row = json::array();
            for (int c = 0; c < cols; ++c) row.push_back(get(r,c));
            values.push_back(std::move(row));
        }
        return values;
    }
    const ggml_tensor * projection_weight(char kind, int & offset) const {
        const auto & layer = model->layers.at(selected_layer);
        const ggml_tensor * w = kind == 'q' ? layer.wq : kind == 'k' ? layer.wk : layer.wv;
        const int kv = selected_head / (heads / kv_heads);
        offset = (kind == 'q' ? selected_head : kv) * dim;
        if (!w && layer.wqkv) {
            w = layer.wqkv;
            offset += kind == 'k' ? heads * dim : kind == 'v' ? (heads + kv_heads) * dim : 0;
        }
        if (!w || offset + dim > w->ne[1]) throw std::runtime_error("Projection weights cannot be inspected for this model");
        return w;
    }
    std::vector<float> weight_row(const ggml_tensor * w, int row) const {
        const auto id=gguf_find_tensor(weight_gguf,w->name);
        if(id<0)throw std::runtime_error("Projection tensor missing from GGUF storage");
        const auto type=gguf_get_tensor_type(weight_gguf,id);
        const auto * traits = ggml_get_type_traits(type);
        std::vector<uint8_t> bytes(ggml_row_size(type, w->ne[0]));
        // CPU kernels may repack weights into an unreadable runtime buffer.
        // Decode the original GGUF row, preserving inference's optimized backend.
        weight_file.clear();weight_file.seekg(gguf_get_data_offset(weight_gguf)+gguf_get_tensor_offset(weight_gguf,id)+row*bytes.size());
        weight_file.read(reinterpret_cast<char *>(bytes.data()),bytes.size());
        if(!weight_file)throw std::runtime_error("Could not read projection weight row");
        std::vector<float> result(w->ne[0]);
        if (type == GGML_TYPE_F32) std::memcpy(result.data(), bytes.data(), bytes.size());
        else if (traits->to_float) traits->to_float(bytes.data(), result.data(), w->ne[0]);
        else throw std::runtime_error("Weight quantization cannot be decoded for inspection");
        for (float value : result) if (!std::isfinite(value)) throw std::runtime_error("Non-finite model weight");
        return result;
    }
    json weights() const {
        if (weight_layer == selected_layer && weight_head == selected_head) return weight_view;
        json result = json::object();
        for (char kind : {'q', 'k', 'v'}) {
            int offset; const auto * w = projection_weight(kind, offset);
            int rows = dim, cols = w->ne[0];
            int er = std::clamp(int(std::round(8 * std::sqrt(rows) / (std::sqrt(rows) + std::sqrt(cols)))), 2, 6), ec = 8 - er;
            auto indices = [](int size, int edge) { std::vector<int> out; for (int i=0; i<size; ++i) if (size<=2*edge+1 || i<edge || i>=size-edge) out.push_back(i); return out; };
            // Retain enough output rows to align with any projection proxy.
            auto ri = indices(rows, 6), ci = indices(cols, ec);
            json values = json::array();
            for (int r : ri) { auto row = weight_row(w, offset+r); json selected=json::array(); for (int c : ci) selected.push_back(row[c]); values.push_back(std::move(selected)); }
            result[std::string("w")+kind] = {{"rows",rows},{"cols",cols},{"row_indices",ri},{"col_indices",ci},{"values",values},{"storage_type",ggml_type_name(gguf_get_tensor_type(weight_gguf,gguf_find_tensor(weight_gguf,w->name)))},{"head",kind=='q'?selected_head:selected_head/(heads/kv_heads)}};
        }
        const auto & l = model->layers.at(selected_layer);
        result["transforms"] = {{"q", std::string("linear") + ((l.wq_b || l.wqkv_b)?" + bias":"") + " → RoPE" + (model->hparams.use_kq_norm?" → head RMSNorm":"")},
            {"k", std::string("linear") + ((l.wk_b || l.wqkv_b)?" + bias":"") + " → RoPE" + (model->hparams.use_kq_norm?" → head RMSNorm":"") + " → F16 cache"},
            {"v", std::string("linear") + ((l.wv_b || l.wqkv_b)?" + bias":"") + " → F16 cache"}};
        result["rope_type"] = int(llama_model_rope_type(model));
        result["rope_dims"] = model->hparams.n_rot(selected_layer);
        result["head_norm"] = model->hparams.use_kq_norm;
        weight_view=result;weight_layer=selected_layer;weight_head=selected_head;return result;
    }
    json view() const {
        json timeline = json::array(), pieces = json::array();
        for (auto id : tokens) pieces.push_back({{"id", id}, {"text", piece(id)}});
        int kv = selected_head / (heads / kv_heads);
        for (const auto & s : steps) {
            int n = s.start + s.count;
            const auto & l = s.layers.at(selected_layer);
            const auto & q = l.at("query");
            const auto & x = l.at("attn_norm"), & a = l.at("kq_soft_max"), & h = l.at("kqv");
            const auto & ck = cache[selected_layer].at("cache_k"), & cv = cache[selected_layer].at("cache_v");
            json matrices;
            matrices["x"] = matrix(x.ne[0], s.count, [&](int r,int c) { return x.at(r,c); });
            matrices["q"] = matrix(dim, s.count, [&](int r,int c) { return q.at(r,c,selected_head); });
            matrices["k"] = matrix(dim, s.count, [&](int r,int c) { return ck.at(r,s.start+c,kv); });
            matrices["v"] = matrix(dim, s.count, [&](int r,int c) { return cv.at(s.start+c,r,kv); });
            matrices["kt"] = matrix(n, dim, [&](int r,int c) { return ck.at(c,r,kv); });
            matrices["scores"] = matrix(n, s.count, [&](int r,int c) { return l.at("scores").at(r,c,selected_head) / std::sqrt(float(dim)); });
            matrices["a"] = matrix(n, s.count, [&](int r,int c) { return a.at(r,c,selected_head); });
            matrices["va"] = matrix(dim, s.count, [&](int r,int c) { return h.at(r,c,selected_head); });
            matrices["kc"] = matrix(dim, n, [&](int r,int c) { return ck.at(r,c,kv); });
            matrices["vc"] = matrix(dim, n, [&](int r,int c) { return cv.at(c,r,kv); });
            timeline.push_back({{"start", s.start}, {"count", s.count}, {"ms", s.ms}, {"matrices", std::move(matrices)}});
        }
        double total_ms = 0; for (const auto & s : steps) total_ms += s.ms;
        return {{"model", info()}, {"weights", weights()}, {"tokens", pieces}, {"steps", timeline}, {"layer", selected_layer},
            {"head", selected_head}, {"kv_head", kv}, {"ended", ended}, {"full", tokens.size() >= size_t(context)},
            {"total_ms", total_ms}, {"cache_prefix_verified", steps.size() > 1},
            {"logical_cache_bytes", tokens.size() * size_t(layers) * kv_heads * dim * 2 * 2}};
    }
    json command(const json & request) {
        auto op = request.at("op").get<std::string>();
        if (op == "load") {
            unload();
            context = request.value("context", 128);
            if (context < 16 || context > 128) throw std::runtime_error("Context must be between 16 and 128 tokens");
            auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
            model = llama_model_load_from_file(request.at("path").get<std::string>().c_str(), mp);
            if (!model) throw std::runtime_error("Unable to load GGUF model");
            if (metadata("general.architecture") != "llama") { unload(); throw std::runtime_error("Only standard Llama architecture is supported"); }
            const auto path=request.at("path").get<std::string>();
            gguf_init_params gp={true,nullptr};weight_gguf=gguf_init_from_file(path.c_str(),gp);weight_file.open(path,std::ios::binary);
            if(!weight_gguf||!weight_file){unload();throw std::runtime_error("Cannot inspect original GGUF weights");}
            vocab = llama_model_get_vocab(model);
            layers = llama_model_n_layer(model); heads = llama_model_n_head(model); kv_heads = llama_model_n_head_kv(model);
            dim = llama_model_n_embd(model) / heads;
            selected_layer = 0; selected_head = 0;
            auto cp = llama_context_default_params();
            cp.n_ctx = context; cp.n_batch = context; cp.n_ubatch = context; cp.n_seq_max = 1;
            cp.n_threads = cp.n_threads_batch = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = cp.type_v = GGML_TYPE_F16;
            cp.offload_kqv = false; cp.op_offload = false;
            cp.cb_eval = callback; cp.cb_eval_user_data = this;
            ctx = llama_init_from_model(model, cp);
            if (!ctx) { unload(); throw std::runtime_error("Not enough memory to create inference context"); }
            // A real smoke test verifies that tensor capture works before accepting the model.
            try { decode(tokenize("Hello")); view(); }
            catch (...) { unload(); throw; }
            llama_memory_clear(llama_get_memory(ctx), true); tokens.clear(); steps.clear();
            return {{"model", info()}, {"smoke_test", "passed"}};
        }
        if (!model || !ctx) throw std::runtime_error("Load a supported model first");
        if (op == "verify_projections") {
            if (steps.empty()) throw std::runtime_error("Prefill a prompt first");
            json report=json::object();
            for (char kind : {'q','k','v'}) {
                int offset; const auto * w=projection_weight(kind,offset);
                double max_error=0, max_relative=0, max_kernel_relative=0;
                for (const auto & s : steps) {
                    const auto & l=s.layers.at(selected_layer);
                    const auto & x=l.at("attn_norm");
                    const std::string name=std::string("raw_")+kind;
                    const auto & raw=l.count(name)?l.at(name):l.at("raw_qkv");
                    int raw_offset=offset;
                    for (int r : {0,dim/2,dim-1}) {
                        auto row=weight_row(w,offset+r);
                        for (int c=0;c<s.count;++c) {
                            double dot=0;for(int d=0;d<w->ne[0];++d)dot+=row[d]*x.at(d,c);
                            double actual=raw.at(raw_offset+r,c),error=std::abs(dot-actual);
                            std::vector<float> input(w->ne[0]),rounded(input.size());for(int d=0;d<w->ne[0];++d)input[d]=x.at(d,c);
                            const auto dot_type=ggml_get_type_traits_cpu(w->type)->vec_dot_type;
                            const auto * cpu=ggml_get_type_traits_cpu(dot_type);const auto * traits=ggml_get_type_traits(dot_type);
                            double quant_dot=dot;
                            if(cpu->from_float&&traits->to_float){
                                std::vector<uint8_t> quant(ggml_row_size(dot_type,input.size()));cpu->from_float(input.data(),quant.data(),input.size());traits->to_float(quant.data(),rounded.data(),input.size());
                                quant_dot=0;for(size_t d=0;d<input.size();++d)quant_dot+=row[d]*rounded[d];
                            }
                            max_error=std::max(max_error,error);max_relative=std::max(max_relative,error/(1+std::abs(actual)));
                            // CPU may choose BLAS (F32 input) or a quantized dot kernel.
                            max_kernel_relative=std::max(max_kernel_relative,std::min(error,std::abs(quant_dot-actual))/(1+std::abs(actual)));
                        }
                    }
                }
                report[std::string(1,kind)]={{"max_error",max_error},{"scaled_error",max_relative},{"kernel_scaled_error",max_kernel_relative}};
            }
            return report;
        }
        if (op == "verify") {
            if (steps.empty()) throw std::runtime_error("Prefill a prompt first");
            auto cp = llama_context_default_params();
            cp.n_ctx = context; cp.n_batch = context; cp.n_ubatch = context; cp.n_seq_max = 1;
            cp.n_threads = cp.n_threads_batch = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = cp.type_v = GGML_TYPE_F16; cp.offload_kqv = false; cp.op_offload = false;
            auto * reference = llama_init_from_model(model, cp);
            if (!reference) throw std::runtime_error("Could not allocate verification context");
            auto batch = llama_batch_get_one(tokens.data(), tokens.size());
            int result = llama_decode(reference, batch);
            if (result != 0) { llama_free(reference); throw std::runtime_error("Reference inference failed"); }
            auto * cached = llama_get_logits_ith(ctx, -1);
            auto * fresh = llama_get_logits_ith(reference, -1);
            int count = llama_vocab_n_tokens(vocab);
            float delta = 0;
            for (int i = 0; i < count; ++i) delta = std::max(delta, std::abs(cached[i] - fresh[i]));
            bool same = std::max_element(cached, cached + count) - cached == std::max_element(fresh, fresh + count) - fresh;
            llama_free(reference);
            return {{"max_logit_difference", delta}, {"same_next_token", same}};
        }
        if (op == "prefill") {
            auto text = request.at("prompt").get<std::string>();
            if (text.empty()) throw std::runtime_error("Enter a prompt");
            auto ids = tokenize(text);
            if (ids.size() >= size_t(context)) throw std::runtime_error("Prompt is too long; leave room for generated tokens");
            llama_memory_clear(llama_get_memory(ctx), true); tokens.clear(); steps.clear(); ended = false;
            decode(ids);
        } else if (op == "step") {
            if (steps.empty()) throw std::runtime_error("Prefill a prompt first");
            if (!ended) {
                auto * sampler = llama_sampler_init_greedy();
                auto id = llama_sampler_sample(sampler, ctx, -1); llama_sampler_free(sampler);
                if (llama_vocab_is_eog(vocab, id)) ended = true;
                else decode({id});
            }
        } else if (op == "view") {
            int l = request.value("layer", selected_layer), h = request.value("head", selected_head);
            if (l < 0 || l >= layers || h < 0 || h >= heads) throw std::runtime_error("Invalid layer or head");
            selected_layer = l; selected_head = h;
        } else throw std::runtime_error("Unknown inference operation");
        return view();
    }
};

int main() {
    llama_log_set([](ggml_log_level, const char * text, void *) { std::cerr << text; }, nullptr);
    ggml_backend_load_all(); llama_backend_init();
    Engine engine;
    std::string line;
    while (std::getline(std::cin, line)) {
        json response;
        try { response = engine.command(json::parse(line)); response["ok"] = true; }
        catch (const std::exception & error) {
            // A failed decode may have partially mutated KV memory. Invalidate it.
            if (engine.ctx) { llama_memory_clear(llama_get_memory(engine.ctx), true); engine.tokens.clear(); engine.steps.clear(); engine.ended = false; }
            response = {{"ok", false}, {"error", error.what()}};
        }
        std::cout << response.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
    }
    return 0;
}
