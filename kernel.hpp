#pragma once

struct uchar4;

void InitializationKernelLauncher(uchar4 *d_out, int width, int height);
void IsingKernelLauncher(uchar4 *d_out, float beta, int width, int height, int iterations_per_draw);
void CleanupIsingKernel();

// void testKernelLauncher(uchar4 *d_out, int width, int height);