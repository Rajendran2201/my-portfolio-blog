# A Few Useful Things to Know about Machine Learning - Explained

## Table of contents

* [Introduction](#Introduction)
	* [What is machine learning?](#Whatismachinelearning)
	* [Where do we apply machine learning?](#Wheredoweapplymachinelearning)
	* [What is classification?](#Whatisclassification)
* [Learning = Representation + Evaluation + Optimization](#LearningRepresentationEvaluationOptimization)
	* [What's the problem here?](#Whatstheproblemhere)
	* [Representation](#Representation)
	* [Evaluation](#Evaluation)
	* [Optimization](#Optimization)
* [It's Generalization that counts](#ItsGeneralizationthatcounts)
* [Data alone is not enough](#Dataaloneisnotenough)
* [Overfitting has many faces](#Overfittinghasmanyfaces)
	* [What is overfitting?](#Whatisoverfitting)
	* [Bias](#Bias)
	* [Variance](#Variance)
	* [How do we not fall into the trap of overfitting?](#Howdowenotfallintothetrapofoverfitting)
* [Intuition fails in high dimensions](#Intuitionfailsinhighdimensions)
	* [Curse of dimensionality](#Curseofdimensionality)
	* [So, what's the problem here?](#Sowhatstheproblemhere)
	* [Is there antidote for this curse?](#Isthereantidoteforthiscurse)
* [Theoretical guarantees are not what they seem](#Theoreticalguaranteesarenotwhattheyseem)
* [Feature engineering is the key](#Featureengineeringisthekey)
	* [What makes a machine learning model successful?](#Whatmakesamachinelearningmodelsuccessful)
	* [One-shot vs Iterative approach](#One-shotvsIterativeapproach)
* [More data beats a cleverer algorithm](#Moredatabeatsaclevereralgorithm)
	* [Do we need more data or better algorithm?](#Doweneedmoredataorbetteralgorithm)
	* [So, why not collect loads of data?](#Sowhynotcollectloadsofdata)
	* [Why does clever algorithms fail?](#Whydoescleveralgorithmsfail)
* [Learn many models, not just one](#Learnmanymodelsnotjustone)
	* [What is ensemble learning?](#Whatisensemblelearning)
	* [Bagging](#Bagging)
	* [Boosting](#Boosting)
	* [Stacking](#Stacking)
* [Simplicity does not imply accuracy](#Simplicitydoesnotimplyaccuracy)
	* [Occam's razor misinterpretation](#Occamsrazormisinterpretation)
* [Representable does not imply learnable](#Representabledoesnotimplylearnable)
* [Correlation does not imply causation](#Correlationdoesnotimplycausation)
	* [What does correlation & causation really mean?](#Whatdoescorrelationcausationreallymean)
	* [Why Correlation ≠ Causation](#WhyCorrelationCausation)
* [Conclusion](#Conclusion)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


This article is an explanation of the research paper I recently studied, called [**A Few Useful Things to Know about Machine Learning**](https://dl.acm.org/doi/10.1145/2347736.2347755) written by [Pedro Domingos](https://homes.cs.washington.edu/~pedrod/). The paper, *"A Few Useful Things to Know about Machine Learning"* is a very famous one and covers most of the fundamental concepts in machine learning.

---



## <a name='Introduction'></a>Introduction 
#### <a name='Whatismachinelearning'></a>What is machine learning?
Machine learning is a type of learning from data. Earlier, systems are constructed by feeding data to them manually which was a tedious process. Thus, it lead to the emergence of machine learning, which automatically learns from the data.

#### <a name='Wheredoweapplymachinelearning'></a>Where do we apply machine learning?
Machine learning has a huge amount of use-cases. Some of them are listed below.
- Spam filters
- Recommendation systems
- Credit scoring
- Fraud detection
- Stock trading
- Web search
- Drug design

There are a lot of machine learning approaches available today. But, for illustration purpose, we'll be looking about the most mature and widely used one: classification.

#### <a name='Whatisclassification'></a>What is classification?
As the name implies, classification is a process of classifying the data. Let say, we have a collection of emails. With classification, we can classify whether an email belong to the spam class or the legitimate class. 
- A *classifier* is a system which takes a vector of discrete/continuous values as input and outputs a discrete value.

![Classifier](https://i.imgur.com/0uLSldE.png)


---
## <a name='LearningRepresentationEvaluationOptimization'></a>Learning = Representation + Evaluation + Optimization

#### <a name='Whatstheproblemhere'></a>What's the problem here?
When a newbie enters into the world of machine learning, it might be overwhelming. Sometimes, more daunting with the exposure to the huge number of learning algorithms available. Though there is no simple recipe to figure out the best appropriate learning algorithm, we'll look into some of the tips to choose the most suitable one later in this article. But before that, we must understand, a machine learning algorithms fundamentally boils down to three components. 

The components are, 
- Representation 
- Evaluation
- Optimization

We'll look into each of them in detail.
#### <a name='Representation'></a>Representation
Learning is a process of making inference from the knowledge / data available by establishing relationships among them. For this, we need the data to be represented in a way that the system can understand. There are various methods of representation available and the choice of representation significantly affects the performance of the learning algorithm. This happens because the representation of data has direct impact on how well the patterns among the data can be established.
##### Hypothesis space
- A model can have any number of functions that can define the relationship between the input and output parameters, which we would call as hypothesis. 
- The collection of all these hypotheses that a learning algorithm will consider during its training is called as **Hypothesis space**.
- A hypothesis $h$ is defined as the specific instance in the hypothesis $H$ space which maps input $X$ to output $Y$.

$$h : X \to Y$$

Where:

- $X$ is the input space 
- $Y$ is the output space 

Our goal is to find the hypothesis $h*$ which posses two properties: 
- It should minimize the error in the training set
- It should generalize well in unseen/new data. i.e. It should provide corrext results even when tested with new/test data.
#### <a name='Evaluation'></a>Evaluation
Evaluation is a process of determining how well does a learning algorithm perform. It is used to compare two or more learning algorithms. It distinguishes good classifiers from the bad ones. This is otherwise called as *objective function* or *scoring function*.

Usually, the evaluation performs on the basis of the model's accuracy, robustness and the degree of generalization. 

#### <a name='Optimization'></a>Optimization
Optimization involves choosing the best classifier among the available. The choice of the optimization technique depends on the evaluation function. The highest scoring classifier is chosen as the optimal classifier. Some of the common optimization techniques involves, gradient descent , genetic algorithms, greedy algorithms and do on.

![](https://i.imgur.com/2mGVIzb.png)

The Table 1 shows the common examples of each of these three components. 

---
Now, we'll see about three famous ways of representations. 
##### $K$-nearest neighbor
- $K$-nearest neighbor is learning algorithm which determines the class that a data belongs to with the help of the classes of the data of the $K$-nearest neighbors in the training dataset.
- Here, $K$ is a user-specified parameter which tells the number of nearest neigbors to consider while making a prediction about a new data-point.
- 
![$K$ Nearest Neigbhor](https://i.imgur.com/a2eAxKx.png)
##### Decision tree
- A decision tree is tree-like representation of data. Each branch of the tree is based on each decision the learning algorithm can possible choose. 
- A decision tree would look like an inverted tree where each node represents a decision.

![](https://i.imgur.com/8Vrq2wD.png)


![Algorithm for decision tree](https://i.imgur.com/2zPEJqJ.png)

##### Support vector machine
- Support vector machine is a learning algorithm which classifies the data by finding a hyper-plane that would separate the data points in the best possible way.
- It is mainly used in high-dimensional spaces.


![Support Vector Machine](https://i.imgur.com/zmZ3Dd1.png)

---
## <a name='ItsGeneralizationthatcounts'></a>It's Generalization that counts

##### What is generalisation?
- Machine learning is all about finding inferences from the given data. Let say, if a model is trained with a data point $X$ to a class $Y$, then if an another data point similar $X$ is given to the model -- it should provide $Y$ as output. 
- If not, it just rote memorisation. We don't ned of these sophisticated models in that case. 
- The goal is to find a general rule between the input values and the output values. This is called as generalisation. 
- If a model is trained with a set of data nad it perform in a certain way. It should also perform similarly when given with some unseen data. 
- Generalisation is a process of making models predict the output accurately even for the unseen data. 
- Thus, the goal of machine learning is to generalise.
##### The need of splitting the data
- In earlier days, a model used to be trained with all the available data and they would randomly provide any one data point from the training data to test the model.
- The modern approach of splitting the data wasn't appreciated during those times. This was because the data scarcity.
- But, this eventually lead to lot of errors or misconceptions as there was no point of reference to evaluate the model properly.
- Now, we have a lot of methods to deal with the data scarcity. So, the splitting of data became very crucial.
- This basically means, whenever you get a dataset to train a model, you have to split the dataset into two parts: training set and test set. 
- You will training the model only with the training data. Then, the model can be evaluated with the help of the test data kept aside.
- This ensures that the training error and the test error would probably be the same.
---
## <a name='Dataaloneisnotenough'></a>Data alone is not enough
##### Why data isn't just enough?
The key component of a learning algorithm is the data it has been fed. Data, being an integral part, is not everything in machine learning. 

Let say, you are trying to build a system which can understand english conversations and provided responses. In such case, no matter how much data you provide, there's going to be an unseen/un-trained sentence from the user. We can't really train a model with the all the possible cases of data. 
##### no free lunch theorem
- When a model encounters a new data point, having no relationship with any of the training data, it no more better than random guessing.
- This is called as **no free lunch theorem**.
- It states, *"no learner can beat random guessing over all possible functions to be learned*.
##### Then, how machine learning is possible?
Luckily, we functions we want to learn in real life have limited dependencies, and limited complexity. The functions are not arbitrary but in contrast, they possess some patterns which makes machine learning possible.
In fact, we use something called **Induction**, which means inferring rules from specific data or observations. Induction requires much less input to produce useful outputs. 

> Induction is a knowledge lever: it turns small amount of input knowledge into large amount of output knowledge.
##### Programming vs learning
Programming is a way of tell a system what and how to do it. It's a lot of work. We have to build everything from the scratch. But learning refers to allowing the system to infer its own rules from the data we provide. Machine learning is no magic, it can't get something out of nothing. What it does it get more from less.

> *Learning is more like farming, which lets nature do most of the work. Farm- ers combine seeds with nutrients to grow crops. Learners combine knowledge with data to grow programs.*

---
## <a name='Overfittinghasmanyfaces'></a>Overfitting has many faces
#### <a name='Whatisoverfitting'></a>What is overfitting?
- When a model learns very specific rules about the training data and fails to generalise the rules, it is called as overfitting.
- We can sense the problem of overfitting when the model works well with the training data but performs poorly with the test data.

![Overfitting](https://i.imgur.com/zGgvVnG.png)

There are many forms of overfitting. We'll try to understand overfitting in terms of bias and variance. 
#### <a name='Bias'></a>Bias 
- Bias refers the learner's tendency to consistently learn the wrong thing.
- In such case, the model fails to understand the underlying pattern in the training data.
- This would eventually lead to underfitting.
#### <a name='Variance'></a>Variance 
- Variance refer's to the learner's tendency to learn random things irrelevant to the required pattern.
- High variance will lead the model to learn unnecessary things (noise) which causes overfitting.

![Bias and variance in dart throwing](https://i.imgur.com/osg063T.png)


The key is finding the sweet spot where the model generalizes well:

- **High Bias** → Underfitting: Not complex enough to capture the underlying data structure.
- **High Variance** → Overfitting: Too complex, capturing noise instead of the true signal.
- **Balanced** → Optimal Generalization: Captures the true data patterns without overfitting.
#### <a name='Howdowenotfallintothetrapofoverfitting'></a>How do we not fall into the trap of overfitting?
We have a lot of techniques to prevent overfitting. There is no single best strategy for this. We'll see about two of them.
##### Regularization 
- Regularization is a technique which add more constraints or penalties to the model to prevent overfitting.
- It helps to improve generalization such that the model performs better even when test with unseen data.
- It avoids training the model with the noise or irrelevant patterns in the training data.
##### Cross-validation
- Cross-validation is a resampling technique used to evaluate the performance ofthe model and helps it to generalize well with the unseen data. 

**Working of cross-validation**
1. The dataset is split into $k$ subsets (folds).
2. The model is trained on $k - 1$  folds and tested on the remaining fold.
3. The process is repeated $k$ -times, with each fold used exactly once as a test set.
4. The performance metrics are averaged over all $k$  folds.

![](https://i.imgur.com/rkZZBQF.png)

---
## <a name='Intuitionfailsinhighdimensions'></a>Intuition fails in high dimensions

#### <a name='Curseofdimensionality'></a>Curse of dimensionality
We have seen about he most important problem in machine learning, which is overfitting. Beyond overfitting, we also have another problem, *"the curse of dimensionality"*.
- Many machine learning models works well with lower-dimensional input. But, when they are given with higher-dimensional input, they perform in a very different way.
- The process of generalisation becomes harder as the dimensionality of the input grows.
#### <a name='Sowhatstheproblemhere'></a>So, what's the problem here?
- In high dimensions, most of the mass of a multivari- ate Gaussian distribution is not near the mean, but in an increasingly distant “shell” around it.
- Most of the volume of a high dimensional orange is in the skin, not the pulp. 
- If a constant number of examples is distributed uniformly in a high-dimensional hypercube, beyond some dimensionality most examples are closer to a face of the hypercube than to their nearest neighbor. 
- And if we approximate a hyper- sphere by inscribing it in a hypercube, in high dimensions almost all the volume of the hypercube is outside the hyper- sphere. 
This is bad news for machine learning, where shapes of one type are often approximated by shapes of another.

> If people could see in high dimensionality, Machine learning would be unnecessary.

#### <a name='Isthereantidoteforthiscurse'></a>Is there antidote for this curse?

Fortunately, most application examples are not spread uniformly throughout he instance space, but they are concentrated on or near a lower-dimensional manifold. This is called as **"blessing of non-uniformity**.

---
## <a name='Theoreticalguaranteesarenotwhattheyseem'></a>Theoretical guarantees are not what they seem
There always exists a difference between the theory and in practice. Similar thing applies for machine learning. In machine learning, we are trying to make the possible inferences form the data we know/have. It gives us some probabilisitic guarantees. One such guarantee is the bound on the number of examples required for a good generalisation.

Though these theoretical guarantees helps us to lead a rightful way, they are not always true. Sometimes, they tells us that a classifier needs an impossible number of examples to generalise better. But, in practice, they don't actually need that much. The learner can generalise well even with far fewer examples than the one calculated in the theoretical guarantee. This implies that these theoretical bounds tells has nothing to relate with the practical performance of the learner.

The bias-variance tradeoff which we've seen the previous section has an effect on it. If learner $A$ is better than learner $B$ given infinite data, then $B$ is often better than $A$ given finite data.

The role of these theoretical guarantees is not as a criterion for practical decisions, but it acts as a source of understanding and driving force for designing algorithms.

> **caveat emptor:** *learning is a complex phenomenon, and just because a learner has a theoretical justification and works in practice doesn't mean the former is the reason for the latter.*
---
## <a name='Featureengineeringisthekey'></a>Feature engineering is the key
#### <a name='Whatmakesamachinelearningmodelsuccessful'></a>What makes a machine learning model successful?
When we try to build machine learning models, some may fail and some may succeed. One of the most important factor that impacts the success is the features we chose. We, now have lot of pre-trained models to do our tasks. But they key here is the feature design. 

Nowadays, making a model learn the data is neither so hard nor time-consuming. But what's hard is the feature selection. We have to choose the features properly to ensure the success of the model.

> But bear in mind that features that look irrelevant in isolation may be relevant in combination

Sometimes, features that may look as orphan might actually be useful at later when related with other features. So, we have to very careful while choosing the features. For examples, let say we have a set of features $(x_1, x_2, x_3, ...., x_n)$ , they might not be as useful as individual. But the $X-OR$ of these features might be relevant.
#### <a name='One-shotvsIterativeapproach'></a>One-shot vs Iterative approach
You cannot figure out the right set of features at one shot. All you have to do is, try and try again until you find the right one. Feature engineering is the most difficult as it is more of domain-specific. Feature engineering is more of what comes from intuition, creativity and art.

---
## <a name='Moredatabeatsaclevereralgorithm'></a>More data beats a cleverer algorithm
#### <a name='Doweneedmoredataorbetteralgorithm'></a>Do we need more data or better algorithm?
- Suppose you’ve constructed the best set of features you can, but the classifiers you’re getting are still not accurate enough. What can you do now? 
- There are two main choices: 
	- design a better learning algorithm, or
	- gather more data 
- Machine learning researchers are mainly concerned with the former, but pragmatically the quickest path to success is often to just get more data. 
- As a rule of thumb, a dumb algorithm with lots and lots of data beats a clever one with modest amounts of it. (After all, machine learning is all about letting data do the heavy lifting.) 
#### <a name='Sowhynotcollectloadsofdata'></a>So, why not collect loads of data?
- This does bring up another problem, however: scalability. 
- In most of computer science, the two main limited resources are time and memory. 
- In machine learning, there is a third one: training data. Which one is the bottleneck has changed from decade to decade. In the 1980’s it tended to be data. Today it is often time. 
- Enormous mountains of data are available, but there is not enough time to process it, so it goes unused. 
- This leads to a paradox: even though in principle more data means that more complex classifiers can be learned, in practice simpler classifiers wind up being used, because complex ones take too long to learn. 
- Part of the answer is to come up with fast ways to learn complex classifiers.
#### <a name='Whydoescleveralgorithmsfail'></a>Why does clever algorithms fail?
We might assume that the clever algorithms would be more beneficial but in practice, they are not as much as we expect. All learning algorithms essentially work by grouping nearby examples into the same class. but the key difference is in the meaning of "nearby".

As a rule, it is always advisable to use the simple learners first. For example, naive Bayes before logistic regression or $k$-nearest neighbor before support vector machines and so on. learning algorithms can be divided into two types: 
- **Parametric learners:** Learners have a fixed representation
- **Non-parametric learners:** the representation grows with the data.
Fixed size learners are beneficial when you have loads of data. Variable size leaners can learn any function but in practice, they may not be able to do it. This can be because of the algorithm or the computational costs. So machine learning projects often wind up having a significant component of learner design, and practitioners need to have some expertise in it
---
## <a name='Learnmanymodelsnotjustone'></a>Learn many models, not just one
Earlier days, with the emergence of many learning algorithms, we tend to try out every possible learner out there and struggling to choose the best one out of them. This was a tedious process, as the performance of the learner changes with each use-case. While people were busy in determining the best-performing learning algorithm, Researchers found another way to boost the performance. It was noticed that the learners perform well in combination than in isolation. This lead to the emergence of ensemble learning.
#### <a name='Whatisensemblelearning'></a>What is ensemble learning?
- Ensemble learning is a technique in machine learning in which two or more weak learners are aggregated together to improve the accuracy of the model. 
- This can be done in three ways, 
	- Bagging 
	- Boosting
	- Stacking
#### <a name='Bagging'></a>Bagging 
- Bagging is an acronym of "Bootstrapped Aggregation".
- In this method, the training data is fed to multiple weak learners to improve the accuracy. 
- The output of this technique can be derived in two ways,
	- Averaging: The outputs of all the weak learners are averaged to find the final output.
	- Max voting: The final output is derived by the majority of the outputs of the weak learner.

![Bagging](https://i.imgur.com/ohQDCnG.png)

#### <a name='Boosting'></a>Boosting 
- Boosting is a method, in which the output of one weak learner is provided as the input of another weaker learner.
- The output of the last weak learner in the chain of weak learners is considered as the final output.

![Boosting](https://i.imgur.com/yKHbTJx.png)

#### <a name='Stacking'></a>Stacking
- Stacking is a technique in which a new training set is built using the weak learners. 
- In this method, a new robust model is built with multiple heterogenous weak models, which we could call as "meta-model".
- The final output is then derived from the meta-model.
![Stacking](https://i.imgur.com/zTg8kKb.png)
---
## <a name='Simplicitydoesnotimplyaccuracy'></a>Simplicity does not imply accuracy
The common misconception is that the simple learners are accurate but in practice, there is no relation between them. We prefer learners that are simple by design and is they are accurate, it's because our choice of design is accurate, not because of the simplicity of the learner. For example, The generalization error of a boosted ensemble continues to improve by adding classifiers even after the training error has reached zero.

#### <a name='Occamsrazormisinterpretation'></a>Occam's razor misinterpretation
- Occam's razor famously states that entities should not be multiplied beyond necessity.
- Many of then have misinterpreted it as given two classifiers with the same training error, the simpler of the two will likely have the lowest test error.
- But, This actually implies that simpler hypotheses should be preferred because simplicity is a virtue in its own right, not because of hypothetical connection with accuracy.

> The accuracy of the model has nothing to do with it's complexity.

---
## <a name='Representabledoesnotimplylearnable'></a>Representable does not imply learnable
Another common misconception is, "Every function can be represented, or approximated arbitrarily closely, using this representation". But, this isn't true.  
- A model is _representable_ if it has the capacity to encode or approximate the target function within its hypothesis space.
- A model is _learnable_ if it can discover the target function from a finite dataset using a given learning algorithm.
Just because a function can be represented does not mean it can be learned. 
For example, standard decision tree learners cannot learn trees with more leaves than there are training examples.
- The answer for "Can it be represented?" is often trivial.
- The real question is "Can it be learned?".
The quest of finding methods for learning deeper representations always remains in the field of machine learning.
---
## <a name='Correlationdoesnotimplycausation'></a>Correlation does not imply causation
#### <a name='Whatdoescorrelationcausationreallymean'></a>What does correlation & causation really mean?
- Correlation measures the statistical relationship or association between two variables
- Causation implies that a change in one variable directly causes a change in another.
#### <a name='WhyCorrelationCausation'></a>Why Correlation ≠ Causation
- **Confounding Variables**: A third factor may influence both variables, creating a false impression of causation.  
    _Example_: Ice cream sales and drowning incidents are correlated, but the confounder is warm weather.
- **Reverse Causation**: The direction of causality might be opposite to what is assumed.  
    _Example_: Increased police presence correlates with higher crime rates, but higher crime may lead to increased police presence.
- **Coincidence**: Some correlations occur purely by chance, especially when analyzing large datasets with many variables.  
    _Example_: The number of Nicolas Cage films released in a year correlates with swimming pool drownings—pure coincidence.
- **Mediating Relationships**: A variable may influence another indirectly through an intermediate variable.  
    _Example_: Education level and income correlate, but education leads to higher skill levels, which in turn lead to higher income.
    
The goal of learning predictive models is to use them as guides to action. Machine learning is usually applied to observational data, where the predictive variables are not under the control of the learner, as opposed to experimental data, where they are. Some learning algorithms can potentially extract causal information from observational data, but their applicability is rather restricted. For example, by intuition, putting diaper next to the beer section will increase sales. But without actually doing the experiment, it is difficult to tell.

---
## <a name='Conclusion'></a>Conclusion

Like any discipline, machine learning has a lot of “folk wisdom” that can be hard to come by, but is crucial for success.

Happy learning! 😁