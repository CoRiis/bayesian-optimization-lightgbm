# Bayesian Optimization: Hyper-parmeter tuning for LightGBM.

In this project, I use Bayesian Optimization (BO) to tune a [LightGBM](https://lightgbm.readthedocs.io/en/latest/). The data set used is of less importance, however, the data set consists of 16 features from 50k posts from the social media Instagram. 

In this project, I test four different acquisation functions
- Probability of improvement
- Expected improvement
- Gaussian Process Upper Confidence Bound (GP-UCB)
- Approximation of GB-UCB

First, I optimize the learning rate, the number of leaves, the feature fraction, and the l2-regularization in a sequantial manner, such that each of them I optimized individually. An example of BO with five initial samples and when 30 iterations can be seen below.

![](learning_rate_prob_improvement.gif)

Lastly, I use BO to tune all the parameters in parallel, which yields the best results. For more results, see the short report: Bayesian Optimization for LightGBM.
