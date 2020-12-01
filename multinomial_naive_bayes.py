import numpy as np


class MultinomialNaiveBayes:
    
    def __init__(self):
        self.feature_probabilities = {}
        self.prior = {}


    def fit(self, X, y, alpha=1):
        X = np.array(X)
        y = np.array(y)

        classes, counts = np.unique(y, return_counts=True)
        self.prior = dict(zip(classes, np.log(counts/y.shape[0])))

        for cl in classes:
            class_sum_vec = X[y==cl].sum(axis=0) 
            added_alpha = (X.shape[1] * alpha )
            class_total_sum = class_sum_vec.sum()
            
            self.feature_probabilities[cl] = np.log((class_sum_vec + alpha) / ( added_alpha + class_total_sum))
        

    def predict_probs(self, X):
        X = np.array(X)
        results = []

        for cl in self.prior:
            res = (self.feature_probabilities[cl] * X).sum(axis=1) + self.prior[cl]
            results.append(res)

        results = np.column_stack(results)
        return results


    def predict(self, X):
        X = np.array(X)
        results = self.predict_probs(X)
        return np.argmax(results, axis=1)


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