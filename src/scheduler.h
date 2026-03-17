#pragma once

#include "anima.h"
#include "tensor.h"
#include <vector>

// Sigma schedule builders
std::vector<float> build_normal_sigmas(int num_steps, int num_train_timesteps, float shift);
std::vector<float> build_simple_sigmas(int num_steps, int num_train_timesteps, float shift);
std::vector<float> build_beta_sigmas(int num_steps, int num_train_timesteps, float shift,
                                      float beta_alpha, float beta_beta);
std::vector<float> build_uniform_sigmas(int num_steps, int num_train_timesteps, float shift);

// SNR shift
float time_snr_shift(float alpha, float t);
