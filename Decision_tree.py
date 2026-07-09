import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

    # feature's threshold used for split. value comes into picture only if node is a leaf, giving us output/prediction
    # hence the second function returning the value when node reaches end point.
    # * is a keyword arguement. what it does is that while calling for anything defined after this, we need to specifically
    # call it out by typing its name. in code we can see that we are writing return Node(value=x) this way
    # it tells that anything before this can be passed in normal way, but after this the value thing must be typed out
    # in leaf node we dont have left right or threshold feature only value, so here we mention that while passing into it


class DecisionTreeFromScratch:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    # minsamplesplit means node is allowed to split only if it has two or more samples in it, if 1 then no split
    # root is the root node which is our pointer that connects 3 node instances together for left right split to root node
    # nfeatures is all the features, total of features

    def _entropy(self, y):
        if len(y) == 0:
            return 0.
        counts = np.bincount(y.astype(int))z
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    # we use y because thats how counts are defined, bincount counts the occurences of all unique values
    
    def _create_split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    # argwhere converts the boolean values after the split into actual row indices giving a 2d matrix which is converted
    # to 1-d or simple array using .flatten

    def _information_gain(self,y,X_column,threshold):
        parent_entropy = self._entropy(y)
        left_idxs,right_idxs = self._create_split(X_column,threshold)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        n_l,n_r = len(left_idxs),len(right_idxs)
        entropy_left,entropy_right = self._entropy(y[left_idxs]),self._entropy(y[right_idxs])
        child_entropy = (n_l/len(y))*entropy_left + (n_r/len(y))*entropy_right
        IG = parent_entropy - child_entropy
        return IG
    # since we are using createsplit to split current node,we need the X_column and threshold arguements for that function
    # if any of the left or right total is zero, using basic maths IG is zero, makes sense because it means entire parent
    # node came back into one of the left or right instead of being split so no useful information gained
    # rest code self explanatory

    def _best_split(self,y,X,feat_idxs):
        best_gain = -1
        split_idx,split_thresh = None,None
        
        for feat in feat_idxs:
            X_column = X[:,feat]
            unique_values = np.unique(X_column)

            for threshold in unique_values:
                gain = self._information_gain(y,X_column,threshold)

                if gain > best_gain:
                    split_thresh = threshold
                    split_idx = feat
                    best_gain = gain
                    
        return split_idx,split_thresh

    # here we are finding best possible split by checking for each feature/column, each possible threshold
    # this is done using 2 for loops, checking each column, then each possible value under them
    # initializing target variables, extracting entire column and unique values for clean comparison, then checking IG
    # by testing for every threshold and updating the threshold, column, gain value

    def _build_tree(self,X,y,depth=0):
        if len(y)==0:
            return None
        n_samples,n_feats = X.shape
        n_labels = len(np.unique(y))
        if (depth>=  self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            most_common_label  = np.bincount(y.astype(int)).argmax()
            return Node(value = most_common_label)

        if self.n_features is None:
            feat_idxs = np.arange(n_feats)
        else:
            feat_idxs = np.random.choice(n_feats, self.n_features, replace = False)

        best_feat,best_thresh = self._best_split(y,X,feat_idxs)

        if best_feat is None:
            most_common_label = np.bincount(y.astype(int)).argmax()
            return Node(value=most_common_label)

        left_idxs,right_idxs = self._create_split(X[:,best_feat],best_thresh)

        left_child = self._build_tree(X[left_idxs],y[left_idxs],depth+1)
        right_child = self._build_tree(X[right_idxs],y[right_idxs],depth+1)

        return Node(left = left_child,right = right_child,threshold = best_thresh,feature = best_feat)

    # depth defind in build tree which updates with the recursive loop, 1st section is the termination condition, makes 
    # sense, like limiting depth, min split samples and n_labels = 1 meaning pure value node that is leaf
    # if number of features to consider (N_FEATURES) is given, then we do the random choice, else we select all
    # if we do not get best feat since maybe there is no IG, can happen when we are selecting limited features, then we may
    # get this, there we again use most common label
    # then we split into left right, and runn the recursive loop to build tree under left and right child.
    # added len(y)==0 safety net because we may encounter case where all values go into one side leaving other empty,
    # and when other child is called into build tree, it recieves empty array and this cant be processed by bincount argmax
    # since cant get max for empty seq. so this safety net

    def fit(self,X,y):
        
        X = np.asarray(X, dtype = float)
        y = np.asarray(y,dtype= int)
        self.root = self._build_tree(X,y)

    def _traverse_tree(self,x,node):
        if node is None:
            return 0
            
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)

    # here we are traversing therough the tree made by fit, and using thosee values to predict, by checking for each X
    # and basically fit has all the information for each node's best splits and we put x, start from root node, and keep 
    # checking till we get single value for it using best splits for each node.
    # each x is the value of each row/sample, and hence by doing x[node.feature] we get the best feature and then value
    # inside that feature for that particular row. picking best feature according to node rule, and taking that value from 
    # the sample we compare it with found best threshold. makes sense now.

    def predict(self,X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_tree(x,self.root) for x in X])
    # since we start from root node, we input self.root

if __name__ == "__main__":
    X_train = np.array([[1.0, 1.5], [0.5, 0.8], [2.2, 1.9], [7.0, 8.5], [9.1, 7.8], [8.2, 9.0]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    
    tree = DecisionTreeFromScratch(min_samples_split=2, max_depth=5)
    tree.fit(X_train, y_train)
    
    X_test = np.array([[0.2, 0.4], [8.8, 8.2]], dtype=float)
    predictions = tree.predict(X_test)
    
    print("\n--- Model Inference Verification ---")
    print("Test Sample 1 Prediction (Expected 0):", predictions[0])
    print("Test Sample 2 Prediction (Expected 1):", predictions[1])