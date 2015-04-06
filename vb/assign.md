Variational Inference for Topic Models
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
1.  Don't submit your topics without looking at it first.  It should have topics
    that make sense.  If not, then there's something wrong with your code.
    Here's an example of some of my topics with the default settings of the
    code (10 topics):

```
==========	0	==========
senat	0.0108455
state	0.0090248
feder	0.0082913
congress	0.00635957
offic	0.00632994
republican	0.00632354
committ	0.00606808
reagan	0.0057845
vote	0.00552983
charg	0.00549152
==========	1	==========
dukaki	0.00959087
campaign	0.0084774
bush	0.00778004
state	0.00767836
presid	0.00694226
democrat	0.00635386
vote	0.00627924
jackson	0.00592178
nomin	0.00544024
bill	0.00529073
==========	2	==========
soviet	0.0248781
gorbachev	0.0125443
union	0.0124296
presid	0.0109329
govern	0.00780666
parti	0.00772773
nation	0.00682819
state	0.00620773
leader	0.00610811
econom	0.00607424
==========	3	==========
court	0.0120246
school	0.00915822
student	0.00705007
work	0.00566194
children	0.0055765
record	0.00425499
show	0.00425071
like	0.00411535
parent	0.00376102
case	0.00371467
==========	4	==========
govern	0.0166687
parti	0.0121295
leader	0.0087007
east	0.00791416
presid	0.0078621
offici	0.00764306
nation	0.00757923
west	0.00748638
polit	0.00742713
countri	0.00740816
==========	5	==========
plane	0.0066307
peopl	0.00654259
report	0.00629012
investig	0.00627247
state	0.00622594
offici	0.00613976
pilot	0.00562474
black	0.00545859
train	0.00525848
airlin	0.00464329
==========	6	==========
million	0.0167177
market	0.0127908
compani	0.0126805
price	0.0124785
billion	0.0122056
trade	0.0113311
rate	0.00996076
stock	0.00992481
dollar	0.00725138
bank	0.00721731
==========	7	==========
polic	0.0211217
peopl	0.0139181
kill	0.0102397
report	0.00923044
fire	0.00898517
citi	0.00735152
mile	0.00697055
offici	0.00662041
area	0.00553878
armi	0.0055314
==========	8	==========
charg	0.007054
test	0.00685947
drug	0.00668647
attorney	0.00550733
case	0.00533975
judg	0.00496342
depart	0.00478532
work	0.00450848
offic	0.00434579
system	0.00416308
==========	9	==========
bush	0.0132038
unit	0.0124587
american	0.0112277
forc	0.00963694
presid	0.00924771
iraq	0.00909439
state	0.00892409
militari	0.0078016
soviet	0.00767411
war	0.00730816
```

(Yours might be slightly different because of random seeds.)
