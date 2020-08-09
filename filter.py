import bag_of_words as bw
import multinomial_naive_bayes as mnb



test = ["dear customer, we are pleased to inform  viagra nigeria  you that the invoice for your ipad if attached to this email, best regards, your apple-team"]


bag = bw.BagWords("en")
bag.compute_matrix()
for new in test:
    bag.add_sentence(new)


X, y = bag.save()

divider = -len(test)
X_train = X[:divider]
X_test = X[divider]


multiNB = mnb.MultinomialNaiveBayes()


multiNB.fit(X_train, y)
print(multiNB.predict([X_test]))
