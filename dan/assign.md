
For gaining a better understanding of deep learning, we're going to be
looking at deep averaging networks.  These are a very simple
framework, but they work well for a variety of tasks and will help
introduce some of the core concepts of using deep learning in
practice.

What's a DAN?
==============

A deep averaging network was designed for text documents.  Each word
has a vector represenation.  We average this vector representation,
multiply it by a matrix, and then create a hidden layer.  If it was a
good idea to do it once, we do it again.  The number of times we do it
is called the depth of the network.

[DAN schematic][dan.pdf]

AdaGrad
==============

When you're doing gradient descent, some dimensions are more important
than others.  We won't be doing vanilla SGD; instead, we'll focus on
particular dimensions.  You should not need to modify the code to do
this.  You will not need to implement this, but we have provided a
class that rescales the dimensions as you do your updates.

Autodifferentiation and GPUs
==============

I'm well aware that all of the cool kids are doing deep learning on
GPUs and with deep learning toolkits.  However, using toolkits makes
it harder to learn what's going on, and using GPUs makes programming
slightly more difficult.  To make everyone's life easier, we're going
to run these algorithms on the CPU.  Our data aren't too big, so it
should be fine.
