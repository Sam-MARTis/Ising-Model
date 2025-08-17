#include "kernel.hpp"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "interactions.hpp"

#define W 512
#define H 512

#define ITERATIONS_PER_DRAW 1


#define TITLE_STRING "Ising Model"
GLuint pbo;
GLuint tex;
struct cudaGraphicsResource *cuda_pbo_resource;



void initPixelBuffer()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W * H * sizeof(GLubyte), 0,
                 GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                 cudaGraphicsMapFlagsWriteDiscard);
}

void render(){
    uchar4* d_out = 0;

    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);

    // kernelLauncher(d_out, camera.x, camera.y, camera.z, W, H, SCREEN_SCALING, 0, NULL, NULL);
    // testKernelLauncher(d_out, W, H);
    IsingKernelLauncher(d_out, beta, W, H, ITERATIONS_PER_DRAW);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    glutPostRedisplay();
}
void initializeState(){
    uchar4* d_out = 0;

    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);

    InitializationKernelLauncher(d_out, W, H);

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

}


void drawTexture()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, H);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(W, H);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(W, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}



void display()
{
    render();
    drawTexture();
    glutSwapBuffers();
    // print("Rendering");
    // glutPostRedisplay();
}

void initGLUT(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(W, H);
    glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
    glewInit();
#endif
}

void exitfunc()
{
    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

int main(int argc, char **argv)
{
    printInstructions();
    initGLUT(&argc, argv);
    gluOrtho2D(0, W, H, 0);
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    initPixelBuffer();
    initializeState();
    glutMainLoop();
    atexit(exitfunc);
    return 0;
}

