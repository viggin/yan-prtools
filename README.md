<h2><font color="red">Contents</font></h2>

The YAN-PRTools matlab toolbox now includes 40 common pattern recognition algorithms:

**Feature processing**

1. *mat2ftvec* : Transform sample matrices to a feature matrix
1. *zscore* : feature normalization
1. *pca* : PCA
1. *kpca* : KPCA
1. *lda* : LDA

**Classification**

1. *lr* : Logistic regression
1. *softmax* : Softmax
1. *svm* : Wrapper of libsvm
1. *rf* : Random forest
1. *knn* : K nearest neighbors
1. *gauss* : Wrapper of Matlab's classify function, including methods like naive Bayes, fitting normal density function, Mahalanobis distance, etc.
1. *boost* : AdaBoost with stump weak classifier
1. *tree* : Wrapper of Matlab's tree classifier
1. *ann* : Wrapper of the artificial neural networks in Matlab
1. *elm* : Basic extreme learning machine

**Regression**

1. *ridge* : Ridge regression
1. *kridge* : Kernel ridge regression
1. *svr* : Wrapper of support vector regression in libsvm
1. *simplefit* : Wrapper to Matlab's basic fitting functions, inncluding least squares, robust fitting, quadratic fitting, etc.
1. *lasso* : Wrapper of Matlab's lasso regression
1. *pls* : Wrapper of Matlab's patial least square regression
1. *step* : Wrapper of Matlab's stepwisefit
1. *rf* : Random forest
1. *ann* : Wrapper of the artificial neural networks in Matlab
1. *elm* : Basic extreme learning machine

**Feature selection**

1. *corr* : Feature ranking based on correlation coefficients (filter method)
1. *fisher* : Feature ranking using Fisher ratio (filter method)
1. *mrmr* : Feature ranking using minimum redundancy maximal relevance (mRMR) (filter method)
1. *single* : Feature ranking based on each single feature's prediction accuracy (wrapper method)
1. *sfs* : Feature selection using sequential forward selection (wrapper method)
1. *ga* : Feature selection using the genetic algorithm in Matlab (wrapper method)
1. *rf* : Feature ranking using random forest (embedded method)
1. *stepwisefit* : Feature selection based on stepwise fitting (embedded method)
1. *boost* : Feature selection using AdaBoost with the stump weak learner (embedded method)
1. *svmrfe_ori* : Feature ranking using SVM-recursive feature elimination (SVM-RFE), the original linear version (embedded method)
1. *svmrfe_ker* : Feature ranking using the kernel version of SVM-RFE (embedded method)

**Representative sample selection (active learning)**

1. *cluster* : Sample selection based on cluster centers
1. *ted* : Transductive experimental design
1. *llr* : Locally linear reconstruction
1. *ks* : Kennard-Stone algorithm

<br>
<h2><font color="orange">Interfaces</font></h2>

**Feature processing**

	[Xnew, model] = ftProc_xxx_tr(X,Y,param) % training
	Xnew = ftProc_xxx_te(model,X) % test
	
**Classification**

	model = classf_softmax_tr(X,Y,param) % training
	[pred,prob] = classf_softmax_te(model,Xtest) % test, return the predicted labels and probabilities (optional)
	
**Regression**

	model = regress_xxx_tr(X,Y,param) % training
	rv = regress_xxx_te(model,Xtest) % test, return the predicted values
	
**Feature selection**

	[ftRank,ftScore] = ftSel_xxx(ft,target,param) % return the feature rank (or subset) and scores (optional)
	
**Representative sample selection (active learning)**

	smpList = smpSel_xxx(X,nSel,param) % return the indices of the selected samples

Please see test.m for sample usages.

Besides, there are three uniform wrappers: ftProc_, classf_, regress_. They accept algorithm name strings as inputs and combines the training and test phase.

<br>
<h2><font color="green">Characteristics</font></h2>

* The training (tr) and test (te) phases are split for feature processing, classification and regression to allow more flexible use. For example, one trained model can be applied multiple times.
* The struct "param" is used to pass parameters to algorithms.
* Default parameters are set clearly at the top of the code, along with the explainations.

In brief, I aimed at three main objectives when developing this toolbox:

* Unified and simple interface;
* Convenient to observe and change algorithm parameters, avoiding tedious parameter setting and checking;
* Extensibile. Simple file structures makes it easier to modify the algorithms.

<br>
<h2><font color="blue">Dependencies</font></h2>

In the toolbox, 20 algorithms are self-implemented, 11 are wrappers or mainly based on Matlab functions, and 9 are wrappers or mainly based on 3rd party toolboxes, which are listed below. They are included in the project, however, you may need to recompile some of them depending on your computer platform.

* SVM and SVR: Chih-Chung Chang and Chih-Jen Lin, libsvm (this toolbox is so famous that you only need to google it)
* RF: Abhishek Jaiantilal, [https://code.google.com/p/randomforest-matlab/]()
* ELM: Qin-Yu Zhu and Guang-Bin Huang, [http://www.ntu.edu.sg/home/egbhuang/elm_random_hidden_nodes.html]()
* mRMR: Hanchuan Peng, [http://www.mathworks.com/matlabcentral/fileexchange/?term=authorid%3A27911]()
* TED: Kai Yu, Jinbo Bi, and Volker Tresp, [http://www.dbs.ifi.lmu.de/~yu_k/ted/]()

Thanks to the authors and MathWorks Inc.! I know that there is so many important algorithms not contained in the toolbox, so everybody is welcomed to contribute new codes! Also, if you find any bug in the codes, please don't hesitate to let me know!

Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com