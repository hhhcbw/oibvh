# oibvh
oibvh tree implementation for paper [Binary Ostensibly-Implicit Trees for Fast Collision Detection](https://www.pure.ed.ac.uk/ws/files/142704991/Binary_Ostensibly_Implicit_CHITALU_DOA17022020_AFV.pdf)

---
# 1. Introduction
This project implements collision detection with oibvh tree including:

1. oibvh build
2. oibvh refit
3. collision broad phase
4. collision narrow phase

---
# 2. Configuration and build
This project is developed using Visual Studio 2022 and CUDA 11 on Windows platform.

dependancies:
- cmake 3.10
- assimp

build:
1. `mkdir out`
2. `cd out`
3. `mkdir build`
2. `cd build`
3. `cmake ../..`

note:
1. need download bunny.obj and add it in objects/

---
# 3. Result
a simple case(red section is collided triangles)
![result](./result.gif)