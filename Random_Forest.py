import numpy as np

class Node:
    def __init__(self,left=None,right=None,feature=None,threshold=None,*,value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value

    def _is_leaf_node(self):
        
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split = 2, n_features=None,max_depth=10):
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.max_depth = max_depth
        self.root = None

    def _entropy(self,y):
        if len(y) == 0:
            return 0
        count = np.bincount(y.astype(int))
        probablities = count/len(y)
        return -np.sum(p * np.log2(p) for p in probablities if p > 0)

    def _create_split(self,X_column,threshold):
        left_idxs,right_idxs = None,None
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs,right_idxs

    def _information_gain(self,X_column,y,threshold):
        parent_entropy = self._entropy(y)
        left_idxs,right_idxs = self._create_split(X_column,threshold)
        n_l,n_r = len(left_idxs),len(right_idxs)
        if n_l ==0 or n_r == 0:
            return 0
        entropy_left,entropy_right = self._entropy(y[left_idxs]),self._entropy(y[right_idxs])
        child_entropy = n_l/len(y) * entropy_left + n_r/len(y) * entropy_right
        IG = parent_entropy-child_entropy
        return IG

    def _best_split(self,X,y,feat_idxs):
        best_gain = 0.0
        best_feat,best_thresh = None,None

        for feat in feat_idxs:
            X_column = X[:,feat]
            unique_values = np.unique(X_column)

            for threshold in unique_values:
                gain = self._information_gain(X_column,y,threshold)
                if gain>best_gain and threshold is not None:
                    best_gain = gain
                    best_thresh = threshold
                    best_feat = feat
                    
        if best_feat is None or best_thresh is None:
            return None, None

        return best_feat,best_thresh

    def _build_tree(self,X,y,depth=0):
        if len(y) == 0:
            return None

        n_samples,n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            most_common_label = np.bincount(y.astype(int)).argmax()
            return Node(value = most_common_label)

        if self.n_features is None:
            feat_idxs = np.arange(n_feats)
        else:
            feat_idxs = np.random.choice(n_feats,self.n_features,replace=False)

        best_feat,best_thresh = self._best_split(X,y,feat_idxs)

        if best_feat is None:
            most_common_label = np.bincount(y,astype(int)).argmax()
            return Node(value=most_common_label)

        left_idxs,right_idxs = self._create_split(X[:,best_feat],best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            most_common_label = np.bincount(y.astype(int)).argmax()
            return Node(value=most_common_label)

        left_child = self._build_tree(X[left_idxs],y[left_idxs],depth+1)
        right_child = self._build_tree(X[right_idxs],y[right_idxs],depth+1)

        return Node(left_child,right_child,best_feat,best_thresh)

    def fit(self,X,y):
        
        X = np.asarray(X, dtype = float)
        y = np.asarray(y,dtype= int)
        self.root = self._build_tree(X,y)

    def _traverse_tree(self,x,node):
        if node is None:
            return 0
            
        if node._is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)

    def predict(self,X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_tree(x,self.root) for x in X])

class RandomForest:
    def __init__(self,n_trees=10,min_samples_split=2,max_depth=10,n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def _bootstrap(self,X,y):
        n_samples = X.shape[0]
        
        idxs = np.random.choice(n_samples,size=n_samples,replace=True)
        return X[idxs],y[idxs]

    # selecting random samples method

    def fit(self,X,y):
        self.trees = []

        for _ in range(self.n_trees):

            tree = DecisionTree(min_samples_split = self.min_samples_split
                                ,max_depth = self.max_depth
                                ,n_features = self.n_features)

            X_sample,y_sample = self._bootstrap(X,y)

            tree.fit(X_sample,y_sample)
            self.trees.append(tree)
            
    # here the .fit is called by tree which is decisiontree's fit, not the randomforest fit, hence its tree.fit

    def _most_common_label(self,y):
        return np.bincount(y.astype(int)).argmax()

    # use this to predict outcome by majority vote

    def predict(self,X):

        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds,0,1)
        predictions = [self._most_common_label(sample_pred) for sample_pred in tree_preds]
        return np.array(predictions)

if __name__ == "__main__":
    # Expanded dataset to give the forest something to chew on
    X_train = np.array([
        [1.0, 1.5], [0.5, 0.8], [2.2, 1.9], 
        [7.0, 8.5], [9.1, 7.8], [8.2, 9.0],
        [1.2, 1.1], [7.8, 8.1], [0.9, 0.9], [8.5, 9.5]
    ], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=int)
    
    X_test = np.array([[0.2, 0.4], [8.8, 8.2], [1.5, 1.8], [7.5, 7.9]], dtype=float)
    
    # --- Test Single Tree ---
    print("--- Training Single Decision Tree ---")
    tree = DecisionTree(min_samples_split=2, max_depth=5)
    tree.fit(X_train, y_train)
    tree_preds = tree.predict(X_test)
    print("Tree Predictions: ", tree_preds)
    
    # --- Test Random Forest ---
    print("\n--- Training Random Forest Ensemble ---")
    # Setting n_features=1 forces column randomness at every single split!
    forest = RandomForest(n_trees=5, max_depth=5, n_features=1)
    forest.fit(X_train, y_train)
    forest_preds = forest.predict(X_test)
    print("Forest Predictions:", forest_preds)

