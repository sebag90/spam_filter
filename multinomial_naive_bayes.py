import numpy as np


class MultinomialNaiveBayes:
    
    def __init__(self):
        self.feature_probs = []
        self.prior = []
        self.classes = []
        self.class_count = []
        self.fitted = False
        self.feature_count = []
        self.n_features = 0
        self.coef = []
        self.intercept = []


    def fit(self, X, y, alpha=1):
        if self.fitted == True:
            self.feature_probs = []
            self.prior = []
            self.class_count = []
            self.classes = []
            self.feature_count = []
            self.n_features = 0
            self.coef = []
            self.intercept = []
        
        X = np.array(X)
        y = np.array(y)
        
        self.n_features = y.shape[0]

        classes, counts = np.unique(y, return_counts=True)
        self.prior = np.log(counts/y.shape[0])

        self.class_count = counts
        self.classes = classes
        
        for cl in self.classes:
            feature_count = X[y==cl].sum(axis=0) 
            
            self.feature_count.append(feature_count)
            
            added_alpha = (X.shape[1] * alpha)
            class_total_sum = feature_count.sum()
            
            self.feature_probs.append(np.log((feature_count + alpha) / (added_alpha + class_total_sum)))
        
        if len(self.classes) == 2:
            self.coef = np.array([self.feature_probs[-1]])
            self.intercept = np.array([self.prior[-1]])
        else:
            self.coef = self.feature_probs
            self.intercept = self.prior
            
        
        self.fitted = True
        

    def predict_probs(self, X):
        X = np.array(X)
        results = []

        for i in range(len(self.classes)):
            res = (self.feature_probs[i] * X).sum(axis=1) + self.prior[i]
            results.append(res)

        results = np.column_stack(results)
        return results


    def predict(self, X):
        X = np.array(X)
        results = self.predict_probs(X)
        return self.classes[np.argmax(results, axis=1)]


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    data = load_iris()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    nb = MultinomialNaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print(confusion_matrix(y_test, y_pred))