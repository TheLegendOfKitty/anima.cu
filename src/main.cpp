#include "pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -d DIR     Model directory (default: Anima/split_files)\n");
    fprintf(stderr, "  -p TEXT    Prompt (default: '1girl, solo, smile, best quality')\n");
    fprintf(stderr, "  -n TEXT    Negative prompt (default: 'worst quality, low quality')\n");
    fprintf(stderr, "  -W WIDTH   Image width (default: 1024)\n");
    fprintf(stderr, "  -H HEIGHT  Image height (default: 1024)\n");
    fprintf(stderr, "  --steps N  Number of denoising steps (default: 30)\n");
    fprintf(stderr, "  --cfg F    Guidance scale (default: 4.0)\n");
    fprintf(stderr, "  --seed N   Random seed (default: 42)\n");
    fprintf(stderr, "  --sampler S  Sampler: euler, euler_a_rf (default: euler)\n");
    fprintf(stderr, "  --schedule S Sigma schedule: beta, simple, normal, uniform (default: beta)\n");
    fprintf(stderr, "  --qwen3-tok PATH  Qwen3 tokenizer.json path\n");
    fprintf(stderr, "  --t5-tok PATH     T5 spiece.model path\n");
    fprintf(stderr, "  -o FILE    Output PNG path (default: output.png)\n");
}

int main(int argc, char** argv) {
    AnimaOptions opts;
    std::string model_dir = "Anima/split_files";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            opts.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            opts.negative_prompt = argv[++i];
        } else if (strcmp(argv[i], "-W") == 0 && i + 1 < argc) {
            opts.width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-H") == 0 && i + 1 < argc) {
            opts.height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            opts.num_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) {
            opts.guidance_scale = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            opts.seed = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--sampler") == 0 && i + 1 < argc) {
            opts.sampler = argv[++i];
        } else if (strcmp(argv[i], "--schedule") == 0 && i + 1 < argc) {
            opts.sigma_schedule = argv[++i];
        } else if (strcmp(argv[i], "--qwen3-tok") == 0 && i + 1 < argc) {
            opts.qwen3_tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--t5-tok") == 0 && i + 1 < argc) {
            opts.t5_tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            opts.output_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("Anima CUDA C++ inference engine\n\n");

    AnimaPipeline pipeline;
    pipeline.load(model_dir, opts);
    if (opts.seed == 88888) { pipeline.vae_decode_from_file(opts, "/tmp/python_denoised_latents_bf16.bin"); } else if (opts.seed == 77777) { pipeline.generate_with_cond(opts, "/tmp/python_pos_cond_bf16.bin", "/tmp/python_neg_cond_bf16.bin"); } else { pipeline.generate(opts); }

    return 0;
}
