#pragma once

#include <iostream>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

float temperature = 300.0f;
const float dt = 1.0f;


void print(const auto& message){
    std::cout<<message<<"\n";
}

void printInstructions(){
    print("W for hotter, S for colder");
}

#define SIGN(x) ((x) < 0 ? -1 : 1)

const float epsilon = 0.0001f;

void hotter(){
    temperature += dt;
}

void colder(){
    temperature -= dt;
}



void keyboard(unsigned char key, int x, int y) {

  if(key=='w') hotter();
  if(key=='s') colder();


  if (key == 27) exit(0);

//   glutPostRedisplay();
}