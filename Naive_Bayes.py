  import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

corpus = [
    "Win a free rolex watch now click here", "Click here to claim your cash prize",
    "Urgent account verification required", "Hey are we still meeting for coffee later",
    "Please review the attached project documentation", "Let's grab lunch tomorrow afternoon"
]
labels = [1, 1, 1, 0, 0, 0] 

X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.3, random_state=42)

# Text processing (TF-IDF) and Classifier packaged together to isolate validation folds
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('nb', MultinomialNB())
])

param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)], # Look at single words and word pairs
    'nb__alpha': [0.1, 0.5, 1.0, 2.0]        # (alpha)
}

grid_search = GridSearchCV(nb_pipeline, param_grid, cv=2, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print("\n--- Production Performance ---")
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))