K-Nearest Neighbors
=

Due: 30. January

Overview
--------

In this homework you'll implement a k-Nearest Neighbor classification
framework to take a drawing of a number and output what number it
corresponds to.

![Samples of Handwritten Digits](mnist.png "MNIST Digits")

This is designed to be a *very easy* homework.  If you're spending a
lot of time on this assignment, you are either:
* not prepared to take the course (i.e., if you're struggling with Python)
* seriously over-thinking the assignment

Most of this assignment will be done by calling libraries that have
already been implemented for you.  If you are implementing
n-dimensional search or a median algorithm, you are generating extra
work for yourself and making yourself vulnerable to errors.

You'll turn in your code on Moodle.  This assignment is worth 15
points.  Other assignments will be worth more.

What you have to do
----

Coding:
1.  (Optional) Store necessary data in the constructor so you can do
    classification later.
1.  Modify the _majority_ function so that it returns the *value*
    associated with the most *indices*.
1.  Modify the _classify_ function so that it finds the closest
    indicies to the query point.
1.  Modify the _confusion matrix_ function to classify examples and
    record which number it got right.

Analysis:
1.  What is the role of the number of training points to accuracy?
1.  What is the role of ~k~ to accuracy?
1.  What numbers get confused with each other most easily?

What to turn in
-

1.  Submit your _knn.py_ file
1.  Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)

Unit Tests
=

I've provided unit tests based on the example that we worked through
in class.  Before running your code on read data, make sure it passes
all of the unit tests.

```
cs244-33-dhcp:knn jbg$ python tests.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.003s

OK
```

Initially, it will fail all of them:
```
cs244-33-dhcp:knn jbg$ python tests.py
FFF
======================================================================
FAIL: test1 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 20, in test1
    self.assertEqual(self.knn[1].classify(self.queries[1]), -1)
AssertionError: 1 != -1

======================================================================
FAIL: test2 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 25, in test2
    self.assertEqual(self.knn[2].classify(self.queries[0]), 1)
AssertionError: -1 != 1

======================================================================
FAIL: test3 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 31, in test3
    self.assertEqual(self.knn[3].classify(self.queries[0]), 1)
AssertionError: -1 != 1

----------------------------------------------------------------------
Ran 3 tests in 0.002s

FAILED (failures=3)
```

Example
-

This is an example of what your code should look like:
```
cs244-33-dhcp:knn jbg$ python knn.py --limit 500
Data limit: 500
Done loading data
100/10000 for confusion matrix
200/10000 for confusion matrix
300/10000 for confusion matrix
400/10000 for confusion matrix
500/10000 for confusion matrix
600/10000 for confusion matrix
700/10000 for confusion matrix
800/10000 for confusion matrix
900/10000 for confusion matrix
1000/10000 for confusion matrix
1100/10000 for confusion matrix
1200/10000 for confusion matrix
1300/10000 for confusion matrix
1400/10000 for confusion matrix
1500/10000 for confusion matrix
1600/10000 for confusion matrix
1700/10000 for confusion matrix
1800/10000 for confusion matrix
1900/10000 for confusion matrix
2000/10000 for confusion matrix
2100/10000 for confusion matrix
2200/10000 for confusion matrix
2300/10000 for confusion matrix
2400/10000 for confusion matrix
2500/10000 for confusion matrix
2600/10000 for confusion matrix
2700/10000 for confusion matrix
2800/10000 for confusion matrix
2900/10000 for confusion matrix
3000/10000 for confusion matrix
3100/10000 for confusion matrix
3200/10000 for confusion matrix
3300/10000 for confusion matrix
3400/10000 for confusion matrix
3500/10000 for confusion matrix
3600/10000 for confusion matrix
3700/10000 for confusion matrix
3800/10000 for confusion matrix
3900/10000 for confusion matrix
4000/10000 for confusion matrix
4100/10000 for confusion matrix
4200/10000 for confusion matrix
4300/10000 for confusion matrix
4400/10000 for confusion matrix
4500/10000 for confusion matrix
4600/10000 for confusion matrix
4700/10000 for confusion matrix
4800/10000 for confusion matrix
4900/10000 for confusion matrix
5000/10000 for confusion matrix
5100/10000 for confusion matrix
5200/10000 for confusion matrix
5300/10000 for confusion matrix
5400/10000 for confusion matrix
5500/10000 for confusion matrix
5600/10000 for confusion matrix
5700/10000 for confusion matrix
5800/10000 for confusion matrix
5900/10000 for confusion matrix
6000/10000 for confusion matrix
6100/10000 for confusion matrix
6200/10000 for confusion matrix
6300/10000 for confusion matrix
6400/10000 for confusion matrix
6500/10000 for confusion matrix
6600/10000 for confusion matrix
6700/10000 for confusion matrix
6800/10000 for confusion matrix
6900/10000 for confusion matrix
7000/10000 for confusion matrix
7100/10000 for confusion matrix
7200/10000 for confusion matrix
7300/10000 for confusion matrix
7400/10000 for confusion matrix
7500/10000 for confusion matrix
7600/10000 for confusion matrix
7700/10000 for confusion matrix
7800/10000 for confusion matrix
7900/10000 for confusion matrix
8000/10000 for confusion matrix
8100/10000 for confusion matrix
8200/10000 for confusion matrix
8300/10000 for confusion matrix
8400/10000 for confusion matrix
8500/10000 for confusion matrix
8600/10000 for confusion matrix
8700/10000 for confusion matrix
8800/10000 for confusion matrix
8900/10000 for confusion matrix
9000/10000 for confusion matrix
9100/10000 for confusion matrix
9200/10000 for confusion matrix
9300/10000 for confusion matrix
9400/10000 for confusion matrix
9500/10000 for confusion matrix
9600/10000 for confusion matrix
9700/10000 for confusion matrix
9800/10000 for confusion matrix
9900/10000 for confusion matrix
10000/10000 for confusion matrix
	0	1	2	3	4	5	6	7	8	9
------------------------------------------------------------------------------------------
0:	910	1	9	1	5	17	26	6	2	14
1:	0	1051	2	2	1	5	0	3	0	0
2:	8	79	741	28	21	12	7	63	21	10
3:	5	17	8	883	4	52	6	10	27	18
4:	0	26	1	0	765	0	7	28	1	155
5:	22	21	1	82	13	660	34	14	25	43
6:	16	34	9	0	28	15	863	1	1	0
7:	1	42	2	1	9	3	0	937	0	95
8:	9	59	10	63	13	49	23	30	696	57
9:	6	8	2	14	60	6	5	54	1	805
Accuracy: 0.831100
```

Hints
-

1.  Don't use all of the data, especially at first.  Use the _limit_
    command line argument (as in the above example).  We'll be using
    this dataset again with techniques that scale better.
1.  Don't reimplement closest point data structures or median
    functions.
1.  Make sure your code actually behaves differently for different
    values of _k_.
1.  _median_ can give answers that are not in the label set (e.g. if _k_ is 2).  That's fine for this assignment.
