# polynet

This application is a neural network based tool that is designed to find polynomial which fits some collection of points.

**NOTE**: usage of neural networks here is quite pointless, but it was work interview assignment rather than my own idea.

## Network's architecture

NN input is a single x value (per example). Its output is a vector of coefficients of the approximated polynomial. 

NN used in this project does not need to be more than 1 layer deep (0 hidden layers), because it will overfit anyway. This single fully-connected layer has linear activation. In the beginning number of units in this single layer is equal to *POLYNOMIAL_DEGREE* + 1 value. *POLYNOMIAL_DEGREE* is an argument of *train* command. It is interpreted as maximal degree of the polynomial which will be modeled by the NN. Number of units in the NN can change during training. More on this in the **Coefficients elimination** section.

Loss function is a sum of 2 things:
- mean of variances of coefficients that are output by the NN in an iteration - used to force NN to output the same coefficients independently of x value on input.
- mean square error between y values computed with output coefficients and target y values - used to force NN to get better at approximating the polynomial.

There is no regularization loss as dataset is big enough to prevent overfitting and coefficients elimination might be treated as a form of regularization. 

## Coefficients elimination

During the training some coefficients come out as unnecessary - they have very small value. These coefficients are eliminated through a reduction of number of units in the NN. 

Elimination attempts take place every iteration after certain loss is reached (*ELIMINATION_ATTEMPT_THRESHOLD* constant). Coefficient is eliminated when mean of its absolute values in the current batch is less than or equal to *COEFF_ELIMINATION_THRESHOLD*. Elimination algorithm tries to eliminate coefficients starting from the most significant one. Elimination attempts stop in an iteration when currently analyzed coefficient does not satisfy elimination criteria. This approach helps avoid accidental eliminations as the true most significant coefficient tends to settle first. Then it will act like a barrier between unnecessary coefficients of a higher degree and coefficients which might have not yet settled.  

## Choosing final coefficients

During the training parameters of the best iteration so far (with the lowest loss) are cached - best network variables and average of the coefficients output in this iteration. After the training phase ends final coefficients are simply denormalized version (denormalization of coefficients cancels out the normalization of input parameters) of this best average of coefficients.

## Usage

#### Training

`./polynomial train POLYNOMIAL_DEGREE PATH_TO_CSV`

- *POLYNOMIAL_DEGREE* - upper bound on the degree of a polynomial which will be approximated.
- *PATH_TO_CSV* - path to the .csv file which contains dataset of points. Application will try to fit polynomial to these points.

**IMPORTANT** - *train* command will save 2 files: *net.npz* and *normalization_params.npz* which will be placed in *data* directory **relative to the commands.py script**. Location can be changed through *NORMALIZATION_PARAMS_PATH* and *NET_PATH* constants in this *commands.py* script.

#### Inference

`./polynomial estimate X`

- *X* - float value of x for which y will be estimated using previously trained neural network.

This command will use both *net.npz* and *normalization_params.npz* files so make sure to generate them with *train* command before using *estimate*. 

### 

## Requirements

- Python 2.7.14 (other versions may work too, but it was not tested),
- numpy 1.14.2 (same as above)
