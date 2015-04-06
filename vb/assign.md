Support Vector Machines
=

Due: 24. April

Overview
--------

In this homework you'll explore the primal and dual representations of support
vector machines.

You'll turn in your code on Moodle.  This assignment is worth 30
points.

What you have to do
----

Coding:

1.  Given a beta parameter and a gamma vector, compute the variational paramter
    *phi* for a given word by implementing the *new phi* function.
1.  Given word counts, update the probability of a word given a topic in the m
    step of variational inference in the *m step* function.

Analysis:

1.  Run a topic model on the supplied AP data (data/ap) and turn the resulting
    topic file (topics.txt)

What to turn in
-

1.  Submit your _lda.py_ file
1.  Submit your _topics.txt_ file

Unit Tests
=

I've provided unit tests based on the example that we worked through in class.
Before running your code on read data, make sure it passes all of the unit
tests.  However, these tests are not exhaustive; passing the tests will not
guarantee a good grade, you should verify yourself that your code is robust and
correct.


Hints
-

1.  Test that you can correctly run your code on the toy data before even trying
    the AP data.
1.  It shouldn't take more than 10 minutes to run on the AP data.  If it's much
    slower than that, you probably did something inefficient in your code.
