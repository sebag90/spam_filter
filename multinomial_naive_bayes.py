import numpy as np


class MultinomialNaiveBayes:
    
    def __init__(self):
        self.feature_probabilities = {}
        self.prior = {}
        self.probabilities = []


    def fit(self, X, y, alpha=1):
        X = np.array(X)
        y = np.array(y)

        classes, counts = np.unique(y, return_counts=True)
        self.prior = dict(zip(classes, counts/y.shape[0]))

        for cl in classes:
            class_sum_vec = X[y==cl].sum(axis=0) + alpha
            added_alpha = (X.shape[1] * alpha )
            class_total_sum = X[y==cl].sum(axis=0).sum()
            
            self.feature_probabilities[cl] = class_sum_vec / ( added_alpha + class_total_sum)
        

    def predict(self, X):
        X = np.array(X)
        results = []

        for cl in self.prior:
            res = self.prior[cl] * (self.feature_probabilities[cl] ** X).prod(axis=1)
            results.append(res)

        results = np.column_stack(results)
        self.probabilities = results
        return np.argmax(results, axis=1)


if __name__ == "__main__":
    X =[[1, 1, 0, 1],
        [1, 2, 2, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 1]]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    nb = MultinomialNaiveBayes()
    nb.fit(X, y)
    print(nb.predict([[0, 0, 1, 4], [1, 1, 0, 0]]))