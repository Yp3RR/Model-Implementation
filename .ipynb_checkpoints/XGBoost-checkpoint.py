  import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X, y = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
imbalance_ratio = neg_count / pos_count

xgb_model = XGBClassifier(
    n_estimators=150,          
    learning_rate=0.05,        # Shrinkage factor applied to each tree step (eta)
    max_depth=5,               
    reg_lambda=2.0,            # L2 (lambda) regularization constant for l2
    reg_alpha=0.1,             # L1 (alpha) regularization constant for l1
    scale_pos_weight=imbalance_ratio,  
    random_state=42
    
)

xgb_model.fit(X_train, y_train)

importances_gain = xgb_model.get_booster().get_score(importance_type='gain')
importances_cover = xgb_model.get_booster().get_score(importance_type='cover')

print("--- Model Training Complete ---")
print("Top Feature Gains: ", dict(list(importances_gain.items())[:3]))
