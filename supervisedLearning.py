import sys
import pandas
import numpy
import datetime
import argparse
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import matplotlib.pyplot as plt
#import pydotplus 

#numpy.set_printoptions(threshold=numpy.nan)
#csvFilename=sys.argv[1]
#print 'Dataset is ::' 
#print dataset
#dataset.to_csv("datasetPandas.csv" )
#datasetInput =   dataset.values[:,1:]
#datasetOutput =   dataset.values[:,0]
#testDataInput  =  testData.values[:,1:]
#testDataOutput =  testData.values[:,0]

#print datasetInput
#print datasetOutput


#print numpy.array_equal(predictedArray, testDataOutput)

#dot_data = tree.export_graphviz(decisionTreeClf, out_file=None) 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#graph.write_pdf("iris.pdf") 

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=numpy.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plotBoostDeviance (clf, X_test ,y_test , params ):
    test_score = numpy.zeros((params['n_estimators'],), dtype=numpy.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(numpy.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(numpy.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
def read_csv_get_dataframe(csvFilename ) :
    print 'Working with csv file::' + csvFilename
    csvData     = pandas.read_csv( csvFilename )
    csvData_flat = pandas.get_dummies(csvData)#/Convert categorical variable into dummy indicator/bool columns
    #csvData = csvData.applymap(lambda x: x.is_numeric())
    #csvData = csvData.applymap(lambda x: ord(x))
    #csvData = csvData.applymap(lambda x: x if type(x) in [int, numpy.int64, float, numpy.float64] else ord(x) )
#if there are any chars replace by number
    return csvData_flat

def rand_sample_dataset(csvData, split_frac ) :
    numTraining = int(split_frac *csvData.shape[0])
    print "Number of training samples:"+str(numTraining)
    dataset = csvData.sample(numTraining )
    testData = csvData.drop(dataset.axes[0] )
    #print "\n Saved train and test dataset to train.csv and test.csv"
    #dataset.to_csv("train.csv" )
    #testData.to_csv("test.csv" )
    return dataset, testData


def getLearnedEstimator (trainX,trainY,testX,testY) :
    if optionTable[option_decisionTreeClassifier]==1:
        estimator = LearndecisionTreeClassifier(trainX,trainY,testX,testY)
        str_algo_type="Decision Tree"
    elif optionTable[option_decisionTreeRegressor] ==1 :
        estimator = LearndecisionTreeRegressor(trainX,trainY,testX,testY)
        str_algo_type="Decision Tree"
    elif optionTable[option_annClassifier]==1:
        estimator=LearnMLPClassifier(trainX,trainY,testX,testY )
        str_algo_type="Artificial Neural Network"
    elif optionTable[option_annRegressor]==1:
        estimator=LearnMLPRegressor(trainX,trainY,testX,testY )
        str_algo_type="Artificial Neural Network"
    elif optionTable[option_boostingClassifier]==1:
        estimator=LearnBoostClassifier(trainX,trainY,testX,testY )
        str_algo_type="Boosting"
    elif optionTable[option_boostingRegressor]==1:
        estimator=LearnBoostRegressor(trainX,trainY,testX,testY )
        str_algo_type="Boosting"
    else:
        print "\n No Algorithm Selected Returning default ANN"
        estimator=LearnMLPClassifier(trainX,trainY,testX,testY )
        str_algo_type="Artificial Neural Network"

    print "\n Got estimator :"+str_algo_type
    return estimator, str_algo_type
    
def supervisedLearn(training_dataset, test_dataset, split_frac ) :
    do_scaling = 1
    option_shuffleSplit=0
    trainX = training_dataset.values[:,1:] # First column is the output to be estimated
    trainY = training_dataset.values[:,0]
# Don't cheat - fit only on training data
# apply same transformation to test data
    testX = test_dataset.values[:,1:]
    testY = test_dataset.values[:,0]
    if do_scaling == 1:
        scaler = StandardScaler()  
        scaler.fit(trainX)  
        trainX = scaler.transform(trainX)  
        testX = scaler.transform(testX)  
    for option_index in range(MIN_ALGO_NUM, MAX_ALGO_NUM):
        optionTable[option_index] = 1
        estimator, str_algo_type= getLearnedEstimator (trainX,trainY,testX,testY)
        if (option_shuffleSplit == 1):
            cv = ShuffleSplit(n_splits=50, test_size=0.7, random_state=0)
        allDatasetX=trainX
        allDatasetY=trainY
        #allDatasetX=numpy.concatenate((trainX,testX ))
        #allDatasetY=numpy.concatenate((trainY,testY ))
        title = "Learning Curves " + str_algo_type
        plot_learning_curve(estimator, title, allDatasetX, allDatasetY ) 
        optionTable[option_index] = 0
    return 



def LearnBoostClassifier(trainX,trainY,testX,testY):
    params = {'n_estimators': 10, 'max_depth': 10,'learning_rate': 0.01}
    if (optionTable[option_adaboost] ) :
        boostClf = ensemble.AdaBoostClassifier(**params)
    else  :
        boostClf = ensemble.GradientBoostingClassifier(**params)

    #TODO Enable the following lines
    boostClf.fit(trainX, trainY)
    #plotBoostDeviance(boostClf, testX,testY , params)
    accuracy = boostClf.score(testX,testY )
    print "\n Boosting accuracy:"+str(accuracy)

    return boostClf

def LearnBoostRegressor(trainX,trainY,testX,testY):
    params = {'n_estimators': 500, 'max_depth': 500,'learning_rate': 0.01, 'loss': 'ls'}
    if (optionTable[option_adaboost] ) :
        boostClf = ensemble.AdaBoostRegressor(**params)
    else:
        boostClf = ensemble.GradientBoostingRegressor(**params)

    boostClf.fit(trainX, trainY)
    plotBoostDeviance(boostClf, testX,testY , params)
    return boostClf

def MLPplot_on_dataset(X, y):
    params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
               'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
               'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 0.2},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'adam', 'learning_rate_init': 0.01}]

    labels = ["constant learning-rate", "constant with momentum",
              "constant with Nesterov's momentum",
              "inv-scaling learning-rate", "inv-scaling with momentum",
              "inv-scaling with Nesterov's momentum", "adam"]

    plot_args = [{'c': 'red', 'linestyle': '-'},
                 {'c': 'green', 'linestyle': '-'},
                 {'c': 'blue', 'linestyle': '-'},
                 {'c': 'red', 'linestyle': '--'},
                 {'c': 'green', 'linestyle': '--'},
                 {'c': 'blue', 'linestyle': '--'},
                 {'c': 'black', 'linestyle': '-'}]
    # for each dataset, plot learning for each learning strategy
    fig, ax = plt.subplots()
    name = "Dataset"
    
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 900

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,hidden_layer_sizes=(50,50,), max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)
    #fig.legend(ax.get_lines(), labels=labels, ncol=3, loc="upper center")


def LearnMLPClassifier(trainX,trainY,testX,testY ):
    #mlpClf = MLPClassifier(hidden_layer_sizes=(10,10,35, 35),solver='sgd',learning_rate_init=0.01,max_iter=500 )
    print "\n Running Loss curve plotter"
    MLPplot_on_dataset(trainX, trainY)

    mlpClf=   MLPClassifier( solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,))
    estimator = mlpClf
    mlpClf = mlpClf.fit(trainX, trainY )
    accuracy = mlpClf.score(testX,testY)
    print "\n MLP Classifier accuracy ="+str(accuracy)
    return estimator

def LearnMLPRegressor(trainX,trainY,testX,testY ):
    #mlpClf = MLPClassifier(hidden_layer_sizes=(10,10,35, 35),solver='sgd',learning_rate_init=0.01,max_iter=500 )

    mlpClf=   MLPRegressor( solver='sgd', hidden_layer_sizes=(50,50,50, 50), max_iter=50000)
    estimator = mlpClf
    mlpClf = mlpClf.fit(trainX, trainY )
    accuracy = mlpClf.score(testX,testY)
    print "\n MLP Regressor  accuracy ="+str(accuracy)
    return estimator

def LearndecisionTreeClassifier(trainX,trainY,testX,testY):
    now = datetime.datetime.now()
    outBaseName="decisiontree_"+str(now.minute)+str(now.second)
    iter_max_depth=50
    last_accuracy=0
    while 1:
        
        #decisionTreeClf = tree.DecisionTreeRegressor()
        decisionTreeClf = tree.DecisionTreeClassifier(max_depth=iter_max_depth)
        decisionTreeClf = decisionTreeClf.fit(trainX, trainY )
        accuracy = decisionTreeClf.score(testX,testY )
        print "\n depth="+str(iter_max_depth)+" accuracy="+str(accuracy)+" actual depth="+str(decisionTreeClf.tree_.max_depth )
        outFileName=outBaseName+"depth_"+str(iter_max_depth)+".dot"
        with open(outFileName, 'w') as f:
            f = tree.export_graphviz(decisionTreeClf, out_file=f, feature_names=list(training_dataset)[1:]  )
        if (accuracy == last_accuracy or decisionTreeClf.tree_.max_depth != iter_max_depth):
            break
        last_accuracy = accuracy
        iter_max_depth=iter_max_depth+1
        break
    return decisionTreeClf       


def LearndecisionTreeRegressor(trainX,trainY,testX,testY):
    now = datetime.datetime.now()
    outBaseName="decisiontree_"+str(now.minute)+str(now.second)
    iter_max_depth=50
    last_accuracy=0
    while 1:
        
        #decisionTreeClf = tree.DecisionTreeRegressor()
        decisionTreeClf = tree.DecisionTreeRegressor(max_depth=iter_max_depth)
        decisionTreeClf = decisionTreeClf.fit(trainX, trainY )
        accuracy = decisionTreeClf.score(testX,testY )
        print "\n depth="+str(iter_max_depth)+" accuracy="+str(accuracy)+" actual depth="+str(decisionTreeClf.tree_.max_depth )
        outFileName=outBaseName+"depth_"+str(iter_max_depth)+".dot"
        with open(outFileName, 'w') as f:
            f = tree.export_graphviz(decisionTreeClf, out_file=f, feature_names=list(training_dataset)[1:], class_names=('100', '90','80','70','60','50','40','30' ) )
        if (accuracy == last_accuracy or decisionTreeClf.tree_.max_depth != iter_max_depth):
            break
        last_accuracy = accuracy
        iter_max_depth=iter_max_depth+1
        break
    return decisionTreeClf       
#print "\n Testing the test set now:::\n"
#print testData.values
#predictedArray =  decisionTreeClf.predict(testDataInput ) 
MIN_ALGO_NUM=0
MAX_ALGO_NUM=3
option_boostingClassifier       =0
option_decisionTreeClassifier   =1
option_annClassifier            =2
option_decisionTreeRegressor    =3
option_annRegressor             =4
option_boostingRegressor        =5
option_adaboost                 =6
optionTable=[0,0,0,0,0,0,0 ]

if __name__ == '__main__' :

#Init End
    parser = argparse.ArgumentParser(description='learning algorithm selector' )
    parser.add_argument('-csv_file_name',dest='csv_file_name', type=str,required=True,  help='Data File in CSV Format')
    parser.add_argument('-regression',dest='regression', type=bool,required=False,default=False,  help='Is this a regression or classification problem')
    args = parser.parse_args()
    csvFileName= args.csv_file_name
    split_frac=0.9
#Init End

    dataframe_fromcsv= read_csv_get_dataframe(csvFileName )
    training_dataset, test_dataset = rand_sample_dataset(dataframe_fromcsv, split_frac )
    supervisedLearn(training_dataset, test_dataset,split_frac )
    plt.show()









#    dataframe_fromcsv= read_csv_get_dataframe(sys.argv[1] )
#    total_experiments=200
#    exp_iter=0
#    incr_granularity=0.02
#    split_frac=0.9
#    #for exp_iter in range(0, total_experiments ):
## Different iterations work with different size of training samples, 
#    while split_frac <=1:
#        print "\n Split fraction::"+str(split_frac)
#        training_dataset, test_dataset = rand_sample_dataset(dataframe_fromcsv, split_frac )
#        split_frac = split_frac+incr_granularity
#        accuracy = superviseLearn(training_dataset, test_dataset,split_frac )
#        break
#        if accuracy == 1 or split_frac >=1:
#            break
#        #print "\n Train Frac: " + split_frac +" Error: "+ error
#        #if error <= 0.0005
#        #    break
#    plt.show()
