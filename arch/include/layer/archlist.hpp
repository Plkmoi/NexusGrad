// =====================
// file: include/ag/ag_all.hpp (umbrella)
// =====================
#pragma once
#include "affine.hpp"
#include "activation.hpp"
#include "attention.hpp"
#include "norm.hpp"
#include "loss.hpp"
#include "embed.hpp"
#include "opinse.hpp"
#include "tokenus.hpp"
#include "ops/optim.hpp"
#include "tokeni.hpp"
static void init_cuda_plugin() {
    try {
        #if defined(_WIN32)
            const char* plugin_path = "./agkernels_cuda.dll";
        #elif defined(__APPLE__)
            const char* plugin_path = "./libagkernels_cuda.dylib";
        #else
            const char* plugin_path = "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so";
        #endif

        std::cout << "Loading GPU plugin from: " << plugin_path << "\n";
        ag::kernels::load_cuda_plugin(plugin_path);
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR LOADING CUDA PLUGIN: " << e.what() << std::endl;
    }
}

static bool _cuda_loaded = (init_cuda_plugin(), true);

