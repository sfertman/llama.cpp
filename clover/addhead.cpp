// Copies a DFlash draft GGUF and adds output.weight: either the source model's output.weight
// quantized to Q4_K (from BF16/F16/F32), or copied as-is when the source type is already quantized.
// usage: addhead draft.gguf source.gguf out.gguf [q4_k|copy]
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s draft.gguf source.gguf out.gguf q4_k|copy\n", argv[0]); return 1; }
    const std::string mode = argv[4];

    ggml_context * ctx_d = nullptr;
    gguf_context * gd = gguf_init_from_file(argv[1], { /*no_alloc*/ false, &ctx_d });
    if (!gd) { fprintf(stderr, "failed to read %s\n", argv[1]); return 1; }
    if (gguf_find_tensor(gd, "output.weight") >= 0) { fprintf(stderr, "draft already has output.weight\n"); return 1; }

    ggml_context * ctx_s = nullptr;
    gguf_context * gs = gguf_init_from_file(argv[2], { /*no_alloc*/ true, &ctx_s });
    if (!gs) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const int64_t ti = gguf_find_tensor(gs, "output.weight");
    if (ti < 0) { fprintf(stderr, "source has no output.weight\n"); return 1; }
    ggml_tensor * src = ggml_get_tensor(ctx_s, "output.weight");
    const size_t src_bytes = ggml_nbytes(src);
    std::vector<uint8_t> raw(src_bytes);
    FILE * f = fopen(argv[2], "rb");
    fseek(f, (long) (gguf_get_data_offset(gs) + gguf_get_tensor_offset(gs, ti)), SEEK_SET);
    if (fread(raw.data(), 1, src_bytes, f) != src_bytes) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    const int64_t n_per_row = src->ne[0], nrows = src->ne[1];
    fprintf(stderr, "source output.weight: %s [%lld, %lld], %.1f MB\n", ggml_type_name(src->type),
            (long long) n_per_row, (long long) nrows, src_bytes / 1e6);

    ggml_type out_type = src->type;
    std::vector<uint8_t> out_data;
    if (mode == "copy") {
        out_data = std::move(raw);
    } else if (mode == "q4_k") {
        out_type = GGML_TYPE_Q4_K;
        std::vector<float> f32((size_t) n_per_row * nrows);
        if (src->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), f32.data(), f32.size());
        } else if (src->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), f32.size());
        } else if (src->type == GGML_TYPE_F32) {
            memcpy(f32.data(), raw.data(), raw.size());
        } else if (ggml_get_type_traits(src->type)->to_float) {
            // quantized source: dequantize row by row
            const size_t rs = ggml_row_size(src->type, n_per_row);
            for (int64_t r = 0; r < nrows; ++r) {
                ggml_get_type_traits(src->type)->to_float(raw.data() + r * rs, f32.data() + r * n_per_row, n_per_row);
            }
        } else {
            fprintf(stderr, "cannot convert %s\n", ggml_type_name(src->type));
            return 1;
        }
        raw.clear(); raw.shrink_to_fit();
        out_data.resize(ggml_row_size(out_type, n_per_row) * nrows);
        ggml_quantize_init(out_type);
        const size_t n = ggml_quantize_chunk(out_type, f32.data(), out_data.data(), 0, nrows, n_per_row, nullptr);
        if (n != out_data.size()) { fprintf(stderr, "unexpected quantized size\n"); return 1; }
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 1;
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx_o = ggml_init(ip);
    ggml_tensor * out = ggml_new_tensor_2d(ctx_o, out_type, n_per_row, nrows);
    ggml_set_name(out, "output.weight");
    out->data = out_data.data();

    gguf_context * go = gguf_init_empty();
    gguf_set_kv(go, gd);
    for (int64_t i = 0; i < gguf_get_n_tensors(gd); ++i) {
        gguf_add_tensor(go, ggml_get_tensor(ctx_d, gguf_get_tensor_name(gd, i)));
    }
    gguf_add_tensor(go, out);
    if (!gguf_write_to_file(go, argv[3], false)) { fprintf(stderr, "write failed\n"); return 1; }
    fprintf(stderr, "wrote %s with output.weight %s (%.1f MB)\n", argv[3], ggml_type_name(out_type), out_data.size() / 1e6);
    return 0;
}
