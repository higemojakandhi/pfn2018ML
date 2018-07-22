# Report
## Problem2
in `src` folder, run `python problem2.py`   
Accuracy: 0.8376623376623377

## Problem3
### run code
in `src` folder, run `python problem3.py`   
Accuracy:    0.11688311688311688   
Accuracy with Baseline Perturbation: 0.8376623376623377    

Furthermore, We compared with different epsilon
in `src` folder, run `python problem3_report.py`

| epsilon  | 0.001  | 0.005  | 0.01  | 0.05   | 0.1   | 0.2 | 0.5 |
|----------|--------|--------|-------|--------|-------|-----|-----|
| Accuracy | 0.8377 | 0.8247 | 0.773 | 0.4935 | 0.117 | 0.0 | 0.0 |


#### Observation
With FGSM, Accuracy Drops from 0.83766 to 0.11688,
However, with baseline perturbation, it did not drop

As larger epsilon is added, accuracy dropped which resulted 0% Accuracy using epsilon=0.2



## Problem4.1
### Experiment Method
Using the same code for Problem3, I made 10 Loops of    
`x = x + e` to get the gradient as well as the prediction at each loop

### Experiment Result   
| Loops    | 1     | 2      | 3       | 4       | 5   | 6   | 7  |
|----------|-------|--------|---------|---------|-----|-----|----|
| Accuracy | 0.117 | 0.0584 | 0.05195 | 0.05195 | ... | ... | .. |
3~10 Loops were All the same.
It can be observed that 0.05195 is the limit of Accuracy that this FGSM can attack

## Problem4.3
### Experiment Method
 1. Get Gradient dLdx from one of the predictive networks (default = param_0.txt)
 2. Get Prediction from 10 Networks
 3. Index that was highly voted is being outputted from `predict()` function
 4. apply `argmax()` to get final prediction

### Experiment Result
Accuracy: 0.34415584415584416

## Problem4.3 Attack!
### Experiment Method
Against Such Defence, I came up with a simple modification
 1. Get ALL Gradient dLdx from 10 networks
 2. Calc ep for all Networks
 3. Add ALL epsilon
 4. Add to Image, `Xhat = X + simga(eps)_10`
 5. Get Prediction from 10 Networks
 3. Index that was highly voted is being outputted from `predict()` function
 4. apply `argmax()` to get final prediction

### Experiment Result
Accuracy: 0.09740259740259741
