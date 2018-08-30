import math

def read_sample_data(filepath): # creates a 2D list of document-lists
    with open(filepath, "r") as f:
        return [line.strip().split("\t") for line in f.read().split("\n")]

def get_vocabulary(filepath): # reads words from vocab file into list
    with open(filepath, "r") as f:
        return f.read().split("\n")

def calculate_priors(documents): # calculates the prior likelihood values for classes in documents
    priors = {}
    classes = {}
    classes["_count_"] = 0

    # count the number of documents of a certain class
    for document in documents:
        if document[1] in classes:
            classes[document[1]] = classes[document[1]] + 1
        else:
            classes[document[1]] = 1
            
    # calculate count of documents in specific class (from above) divided by total number of documents
    for clazz, count in classes.items():
        if clazz is not "_count_":
            priors[clazz] = count / len(documents)

    return priors

def baggify_documents(documents): # combines all documents into one bag of words for each class
    collected = {}

    # collect class strings
    for document in documents:
        if document[1] in collected:
            collected[document[1]] = collected[document[1]] + " " + document[2]
        else:
            collected[document[1]] = document[2]
            
    # split class strings into individual tokens
    for clazz, body in collected.items():
        collected[clazz] = body.split()
        
    return collected

def word_given_class(word, clazz, bagged_document, vocabulary): # calculates the probability of a word given the class
    frequency = bagged_document[clazz].count(word)
    return (frequency + 1) / (len(bagged_document[clazz]) + len(vocabulary)) # + 1 and + len() for laplace smoothing

def process(): # main function which calls all other functions
    
    vocabulary = get_vocabulary("sampleTrain.vocab.txt")    # read vocabulary into memory
    training_data = read_sample_data("sampleTrain.txt")     # read training documents into memory
    testing_data = read_sample_data("sampleTest.txt")       # read testing documents into memory

    priors = calculate_priors(training_data)                # calculate prior values
    bagged_training = baggify_documents(training_data)      # baggify training documents

    print("Prior probabilities")
    print()
    for clazz, prob in priors.items():
        print("Class " + str(clazz) + " = " + str(prob))

    print("-------------------")
    print("Feature likelihoods")
    
    for word in vocabulary:
        print()
        print("Word: " + word)
        print("Class 0: " + str(word_given_class(word, "0", bagged_training, vocabulary)))
        print("Class 1: " + str(word_given_class(word, "1", bagged_training, vocabulary)))
        
        
    print("-------------------")
    print("Predictions on test data")

    accuracy = 0 # initialise accuracy to all incorrect
    
    for document in testing_data:
        
        predictions = [ # add prior values, first part of p(c) p(w|c)
            priors["0"], priors["1"]
        ]

        # add the likelihoods for each word in each class
        for token in document[2].split():
            if token in vocabulary: # allows further smoothing by removing frequent words
                # log numbers for scalability
                predictions[0] += math.log(word_given_class(token, "0", bagged_training, vocabulary))
                predictions[1] += math.log(word_given_class(token, "1", bagged_training, vocabulary))

        # find result of predictive calculations
        result = "0" if predictions[0] >= predictions[1] else "1"

        # update accuracy
        if result == document[1]:
            accuracy += 1 / len(testing_data)

        # alert user to prediction results 
        print(document[0] + " = " + result)

    print("-------------------")
    # alert user to accuracy of process
    print("Accuracy on test data = " + str(accuracy))

if __name__ == "__main__":
    process()    
