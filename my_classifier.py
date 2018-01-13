from numpy import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from skmultiflow.core.utils.data_structures import InstanceWindow, FastBuffer
from skmultiflow.core.utils.utils import *

class BatchClassifier:

    def __init__(self, window_size=100, max_models=10):
        self.H = []
        self.h = None
        self.window_size = window_size
        self.window = InstanceWindow(max_size=window_size, dtype=float)
        self.num_models = max_models
        # TODO
        return
    
    def partial_fit(self, X, y=None, classes=None):
        # Update window with new data
        r, c = get_dimensions(X)
        
        if self.window is None:
            self.window = InstanceWindow(max_size=self.window_size)            

        for i in range(r):
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            # If window is full, create and train new Decision Tree
            if self.window._num_samples == self.window_size:
                self.h = DecisionTreeClassifier()
                self.h.fit(self.window.get_attributes_matrix(),
                           self.window.get_targets_matrix())
                # Add new Decision Tree to model set
                self._add_to_buffer(self.h)
                # Clear window
                self.window = InstanceWindow(max_size=self.window_size, dtype=float)            
            return
        return
    
        return self
   
    def predict(self, X):
        N,D = X.shape

        # Check there is at least a Decision Tree fitted
        if len(self.H) == 0 :
#            print('Returning zeros, no model yet')
            return zeros(N)
        
        maj = np.argmax(self._predict_proba(X), axis=1)
#        print('Returning predictions ' + str(maj))
        return maj
    
    def _predict_proba(self, X):
        avg = np.average(np.asarray([clf.predict_proba(X) for clf in self.H]),
                         axis=0)
        return avg
    
    def _add_to_buffer(self, item):
        if len(self.H) == self.num_models:
           self.H.pop(0)
        self.H.append(item)
        return self