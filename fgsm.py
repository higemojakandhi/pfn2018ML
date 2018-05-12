### Objective
# Calc Accurary of Prediction Y from Xhat,
# using current Prediction Weights W1,W2,W3,b1,b2,b3
# Xhat = X + E

# Where E = E0*sign(dLdX)     E0=0.1, could be any
# Where L = -log*f(x) = -y_teacher + log(sigma(exp(y)))
# つまり t番目(正解のidx)の予測値 -yt と log
# Where sigma is from 1<t<C, i guess 1~15?

# dLdY = -[0 0 ... 1 ... 0] + sima(y)
# dLdh2 = W3' * dLdY
# ...
# ..
# dLdX

## Compute E=E0*sign(dLdX)
# add E to X to get Xhat
# save file as new pgm Files

# run prediction test once again
