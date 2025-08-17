#pragma once

#include <iostream>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

float beta = 0.3f;
const float db = 0.02f;


void print(const auto& message){
    std::cout<<message<<"\n";
}

void printInstructions(){
    print("W for hotter, S for colder");
}

#define SIGN(x) ((x) < 0 ? -1 : 1)

const float epsilon = 0.0001f;

void hotter(){
    beta *= (1+db);
    print("Beta: " + std::to_string(beta));
    // beta = MIN(1.0f, beta);
}

void colder(){
    beta *= (1-db);
    print("Beta: " + std::to_string(beta));
    // beta = MAX(0.0f, beta);
}



void keyboard(unsigned char key, int x, int y) {

  if(key=='w') hotter();
  if(key=='s') colder();


  if (key == 27) exit(0);

//   glutPostRedisplay();
}