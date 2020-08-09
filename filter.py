import bag_of_words as bw
import multinomial_naive_bayes as mnb
import json


test = ["dear customer, we are pleased to inform  viagra nigeria  you that the invoice for your ipad if attached to this email, best regards, your apple-team", 
        "dear customer, we are pleased to inform you that the invoice for your ipad if attached to this email, best regards, your apple-team",
        "enlarge your penis in no time with this ipad"]


bag = bw.BagWords("en")
bag.compute_matrix()

for new in test:
    bag.add_sentence(new)


X, y = bag.save()
X_train = X[:-len(test)]
X_test = X[len(X_train):]


multiNB = mnb.MultinomialNaiveBayes()
multiNB.fit(X_train, y)
results = multiNB.predict(X_test)

output = {"spam": [],
          "not spam": []}

for i in range(len(test)):
    if results[i] == 0:
        output["not spam"].append(test[i])
    else:
        output["spam"].append(test[i])


with open("results.json", "w") as file:
        json.dump(output, file, indent=4)
