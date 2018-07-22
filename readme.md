# README

## Release Date
2018/07/23
It is uploaded after the deadline. 

### before
![before](https://user-images.githubusercontent.com/11141442/43048506-bd14c496-8e23-11e8-9503-e53fb4888fcc.png)

### after adding adversal input
![after](https://user-images.githubusercontent.com/11141442/43048519-e954fd00-8e23-11e8-8b56-c7220ea896a8.png)

## How to run source files
### Problem1
Please Check **Array.py**

### Problem2
run `python problem2.py`   
it will print the Accuracy Percentage on your terminal

### Problem3
run `python problem3.py` for eps0=0.1 AND baseline perturbation and save image files

run `python problem3_report.py` for eps0 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

### Problem4
run `python problem4_1.py` for FGSM Loop (Problem4.1)   
I set 10 loops for `xhat = x + eps`   
It will print out Accuracy for each Loop

run `python prolbem4_3.py` for DEFENDING from FGSM   
run `python prolbem4_3_attack.py` for ATTACKING Modified Predictive Model


# Source File Explanation
`Array.py`:       NumPy Like Vector and Matrix(Limited) Calculation   
`layers.py`:      Relu, Affine, Softmax Layers are defined   
`functions.py`:   vector calculations such as `softmax()` are defined   
`fileloader.py`:  load and save pgm & Parameters   
`PretrainedThreeLayerNet.py`:   Three Layer Net with Relu, Affine, Softmax   


# REPORT
Please Check `report.md` in the same directory
