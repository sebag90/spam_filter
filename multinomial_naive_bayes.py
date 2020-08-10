import numpy as np


class MultinomialNaiveBayes:
    
    def __init__(self):
        self.feature_probs_0 = []
        self.feature_probs_1 = []
        self.prior_0 = 0
        self.prior_1 = 0


    def fit(self, X, y):
        X = np.asarray(X)
        probs = {0 : [0]* len(X[0]),
                 1 : [0]* len(X[0]),
                 "sum0" : len(X[0]),
                 "sum1" : len(X[0])}


        # calculate number of occurencies per class and total sum of occurencies for each label
        for i in range(len(X)):
            for j in range(len(X[i])):
            
                if y[i] == 1:
                    probs[1][j] += X[i][j]
                    probs["sum1"] += X[i][j]
                else:
                    probs[0][j] += X[i][j]
                    probs["sum0"] += X[i][j]
        
        
        # calculate feature probability
        for i in probs[0]:
            self.feature_probs_0.append((i + 1)/probs["sum0"])
        for i in probs[1]:
            self.feature_probs_1.append((i+1)/probs["sum1"])



        # calculate prior probability for each class
        for i in y:
            if i == 0:
                self.prior_0 += 1
            else:
                self.prior_1 += 1
                
        self.prior_1 = self.prior_1 / len(y)
        self.prior_0 = self.prior_0 / len(y)




    def predict(self, X):
        X = np.asarray(X)
        results = []
        for x in X:
            ph = self.prior_0
            ps = self.prior_1
            
            for i in range(len(x)):
                if x[i] != 0:
                    ph = ph * (self.feature_probs_0[i] ** x[i])
                    ps = ps * (self.feature_probs_1[i] ** x[i])
        
            if ph > ps:
                results.append(0)
            else:
                results.append(1)

        return results


if __name__ == "__main__":
    X = [[1, 1, 0, 1],
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

    
