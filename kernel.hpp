#pragma once

struct uchar4;

void kernelLauncher(uchar4 *d_out, float temperature, int width, int height, int iterations_per_draw);

// void testKernelLauncher(uchar4 *d_out, int width, int height);