## Answering Questions about Science

To make sure that folks have a good project to work on, I make sure to provide a project (with data) that folks can work on even if they are not inspired with a great project idea.

This is a project that I care about.  One project that I work on is to help [computers answer questions](https://www.youtube.com/embed/kTXJCEvCDYk).  

As you can see from the video, our system does not answer science questions very well.  That's where I need your help.

The setup is relatively simple.  There are a bunch of questions of the form:

> This phenomenon occurs twice as quickly in the Moran model as in the Wright-Fisher model.

You have four options, one of which is correct:

1. Genetic drift
2. Hamiltonian (quantum mechanics)
3. Georg Wilhelm Friedrich Hegel
4. Group (mathematics)

In the training data, you know that the correct answer is the the first answer (Genetic drift).  Your goal is to use machine learning to learn how to answer these questions on test questions which have the same form.

## AI2 Science Challenge

This is improtant for me, but other folks are working on similar challenges.  There's a big [Kaggle challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge) trying to answer science questions.  We'll use the same data format that they're using.

## How our Data are Different, where to submit

Unlike the AI2 challenge, all of our questions will have answers that are entities (i.e., things that have Wikipedia pages).  I suspect that this will make it quite a bit easier.  So you may want to try to do our challenge first; but if your method generalizes to both, then that's great!

I've set up a [Kaggle site](http://inclass.kaggle.com/c/quiz-bowl-science) in the same spirit.

We may be able to add additional data in November.

## What happens if I do well?

You get a good grade, my thanks, and your ideas will be integrated into the next version of our system.  You can also keep working and try to win the big Kaggle competition in February.

## Data

We provide the following files:
* [sci_train.csv](sci_train.csv) - the training set
* [sci_test.csv](sci_test.csv) - the test set
* [sci_sample.csv](sci_sample.csv) - a sample submission file in the correct format

With the following fields:
* id - The ID of the question
* correctAnswer - The correct answer of the question (based on columns)
* answerA/B/C/D - A candidate answer to the question (always a Wikipedia page)

## FAQ

*Do I have to do both challenges?*

No, you can choose to focus on one or othe other.  

*Do you have any clues/code?*

You can take a look at [our code](https://github.com/pinafore/qb); it might give you ideas.  However, it doesn't do anything science specific, which will likely be helpful.

*Can I use external data?*

Yes, in fact I don't think it's possible to succeed without using external data.  However, you're not allowed to use quiz bowl questions to help you answer these questions (beyond the data I provide).
