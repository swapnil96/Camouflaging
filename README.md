# Camouflaging
Camouflage a cube in a 3D-scene

There is a [paper](http://andrewowens.com/papers/2014_camo.pdf) on the same topic from MIT CSAIL.

We tried to implement the paper along with a user interface to put the object in the scene.

## Objective
To hide an Object from different viewpoints

## Inputs

### 3D Scene
About 10-20 images of the scene are captured before any object is placed.

First to make the 3D reconstruction we have used the state-of-the-art [Bundler](http://www.cs.cornell.edu/~snavely/bundler/).

It is used for estimating Structure from Motion (SfM) for Unordered Image Collections.

### 3D object (Cube)
A cube of small dimension (not sure about the exact range) to be placed in the scene.

For this we used [Blender](https://www.blender.org/) to create the 3D scene.

Then we import the **bundle.out** file to blender. 

Blender doesn't support Bundler file out of the box but I found an amazing script to do so. Its included in the /src/Blender directory.

## Output
Texture to be assigned to each of the faces of the cube. So that if we project that texture to that cube surface it will make it blend with the background more accurately.



## References
[1] Camouflaging an Object from Many Viewpoints
[Reference](http://camouflage.csail.mit.edu/)
