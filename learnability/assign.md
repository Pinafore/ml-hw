Learnability
=

Due: 2. Oct (50 Points)

Overview
--------

For this homework, we will not be building anything practical.  Instead, we will
implement algorithms that demonstrate and measure the importance of having
hypothesis classes that aren't too powerful.

Implementation
-

Complete the following functions:
* rademacher: Classify.correlation
* rademacher: origin\_plane\_hypotheses
* rademacher: axis\_aligned\_hypotheses
* rademacher: rademacher\_estimate
* vc\_sin: train\_sin\_classifier

What you can assume
------

You can assume any inequality in the books appendicies or anything we *proved*
in class.  If you have doubts, ask on Piazza.

Analysis
-

In your discussion file:
* argue about an ordering of hypothesis classes in terms of complexity:
  hyperplanes through the origin, arbitrary hyperplanes, and axis-aligned
  rectangles (you can use your experiments as a guide, but simply reporting
  those numbers is not sufficient; you must make a mathematical argument)
* prove that your frequency correctly classifies any training set (up to
  floating point precision on the computer).

What to turn in
------

Turn in your completed python files
* rademacher.py
* vc_sin.py

As well as a discussion file
* discussion.pdf

Hints
------
1.  Feel free to use _bst.py_ for finding points in a range
1.  You may want to use trigonometric functions for the hyperplane function
1.  Do not make your code too slow; you will not get full credits if your code
    does not complete in reasonable time
