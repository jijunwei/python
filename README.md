>>> import numpy
>>> import theano.tensor as T
>>> from theano import function
>>> x = T.dscalar('x')
>>> y = T.dscalar('y')
>>> z = x + y
>>> f = function([x, y], z)
And now that we’ve created our function we can use it:
>>> f(2, 3)
array(5.0)
>>> numpy.allclose(f(16.3, 12.1), 28.4)
True

>>> type(x)
<class 'theano.tensor.var.TensorVariable'>
>>> x.type
TensorType(float64, scalar)
>>> T.dscalar
TensorType(float64, scalar)
>>> x.type is T.dscalar
True
>>> 


>>> from theano import pp
>>> print(pp(z))
(x + y)
>>> 


>>> x = T.dmatrix('x')
>>> y = T.dmatrix('y')
>>> z = x + y
>>> f = function([x, y], z)
dmatrix is the Type for matrices of doubles. Then we can use our new function on 2D arrays:
>>> f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
array([[ 11.,  22.],
       [ 33.,  44.]])

The variable is a NumPy array. We can also use NumPy arrays directly as inputs:
>>> import numpy
>>> f(numpy.array([[1, 2], [3, 4]]), numpy.array([[10, 20], [30, 40]]))
array([[ 11.,  22.],
       [ 33.,  44.]])


The following types are available:
* byte: bscalar, bvector, bmatrix, brow, bcol, btensor3, btensor4, btensor5, btensor6, btensor7
* 16-bit integers: wscalar, wvector, wmatrix, wrow, wcol, wtensor3, wtensor4, wtensor5, wtensor6, wtensor7
* 32-bit integers: iscalar, ivector, imatrix, irow, icol, itensor3, itensor4, itensor5, itensor6, itensor7
* 64-bit integers: lscalar, lvector, lmatrix, lrow, lcol, ltensor3, ltensor4, ltensor5, ltensor6, ltensor7
* float: fscalar, fvector, fmatrix, frow, fcol, ftensor3, ftensor4, ftensor5, ftensor6, ftensor7
* double: dscalar, dvector, dmatrix, drow, dcol, dtensor3, dtensor4, dtensor5, dtensor6, dtensor7
* complex: cscalar, cvector, cmatrix, crow, ccol, ctensor3, ctensor4, ctensor5, ctensor6, ctensor7



Exercise
import theano
a = theano.tensor.vector() # declare variable
out = a + a ** 10               # build symbolic expression
f = theano.function([a], out)   # compile function
print(f([0, 1, 2]))
[    0.     2.  1026.]
Modify and execute this code to compute this expression: a ** 2 + b ** 2 + 2 * a * b.




