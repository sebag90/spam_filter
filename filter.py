import bag_of_words as bw
import multinomial_naive_bayes as mnb
import json
import os



def retrieve_texts(path):
    texts = []
    for filename in os.listdir(path):
        path_to_file = f"{path}/{filename}"

        with open(path_to_file, "r") as file:
            my_string = file.read()
            texts.append(my_string)
    return texts
                

def main():

    ham = retrieve_texts("input/ham")
    spam = retrieve_texts("input/spam")
    y_train = [0 for i in ham] + [1 for j in spam]
    train_data = ham + spam

    
    bag = bw.BagWords("en")
    
    for text in train_data:
        bag.add_sentence(text)

    to_classify = retrieve_texts("input/to_classify")

    for new in to_classify:
        bag.add_sentence(new)
       
    bag.compute_matrix()

    X_train = bag.matrix[:len(y_train)]
    X = bag.matrix[len(y_train):]
    
    nb = mnb.MultinomialNaiveBayes()

    nb.fit(X_train, y_train)
    results = nb.predict(X)
    

    output = {"spam": [],
              "not spam": []}


    for text, label in zip(to_classify, results):
        if label == 0:
            output["not spam"].append(text)
        else:
            output["spam"].append(text)

    with open("results.json", "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()