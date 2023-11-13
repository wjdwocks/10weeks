import numpy as np
from sklearn import (datasets, naive_bayes, metrics, svm, tree, model_selection)
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

if __name__ == '__main__':
    # Load a dataset
    wdbc = datasets.load_breast_cancer()
    
    
    # Train a model # , min_samples_split=10, min_samples_leaf=6, criterion='entropy'
    # t_model = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2) # TODO
    s_model = svm.SVC()
    # nb_model = naive_bayes.GaussianNB()   
    f_model = RandomForestClassifier(n_estimators=80)
    g_model = GradientBoostingClassifier(n_estimators=80)
    # cv_results = model_selection.cross_validate(g_model, wdbc.data, wdbc.target, cv=5, return_train_score=True)
    ensemble_model = VotingClassifier(estimators=[   ('support_vector', f_model),    ('gradient_boosting', g_model)    ], voting='soft')

    
    cv_results = cross_validate(ensemble_model, wdbc.data, wdbc.target, cv=5, return_train_score=True)
    
    # Evaluate the model
    
    
    acc_train = np.mean(cv_results['train_score'])
    acc_test = np.mean(cv_results['test_score'])
    print(f'* Accuracy @ training data: {acc_train:.3f}')
    print(f'* Accuracy @ test data: {acc_test:.3f}')
    print(f'* Your score: {max(10 + 100 * (acc_test - 0.9), 0):.0f}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    