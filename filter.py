import bag_of_words as bw
import multinomial_naive_bayes as mnb
import json
import os


# TODO: restructure, read files here and pass sentences to bag


def retrieve_texts():
    emails = []
    for filename in os.listdir("./input/to_classify"):
        path = "./input/to_classify/" + filename
        with open(path, "r") as file:
            my_string = file.read()
            emails.append(my_string)
    return emails
                

def main():
    test = retrieve_texts()

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


    with open("results.json", "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()