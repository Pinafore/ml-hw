Support Vector Machines
=

Due: 6. March

Overview
--------

In this homework you'll explore the primal and dual representations of support
vector machines.

You'll turn in your code on Moodle.  This assignment is worth 25
points.

What you have to do
----

Coding:

1.  Given a weight vector, implement the *find support* function that returns the indices of the support vectors.
1.  Given a weight vector, implement the *find slack* function that returns the indices of the slack vectors.
1.  Given the alpha dual parameterization, implement the *weight vector* function that returns the corresponding weight vector.

Analysis:

1.  Use the scikit implementation of support vector machines to train a classifier to distinguish 3's from 8's.  (Use the MNIST data from the KNN homework.)
1.  Try at least five values of the regularization parameter _C_ and at least two kernels.
1.  Give examples of support vectors with a linear kernel.

What to turn in
-

1.  Submit your _svm.py_ file
1.  Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)

Unit Tests
=

I've provided unit tests based on the example that we worked through in class.
Before running your code on read data, make sure it passes all of the unit
tests.  However, these tests are not exhaustive; passing the tests will not
guarantee a good grade, you should verify yourself that your code is robust and
correct.


Hints
-

1.  Don't use all of the data, especially at first.  You'll want to implement a _limit_ functionality.
