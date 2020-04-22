# Starter code for CS 165B HW2 Spring 2019
import numpy as np

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    #grab the size of the training data input for each of the classes
    num_A_train = training_input[0][1]
    num_B_train = training_input[0][2]
    num_C_train = training_input[0][3]
    #remove the information that we used to find the size of the classes and segregate each of the
    #classes into their own numpy array
    training_input.remove(training_input[0])
    training = np.array(training_input)
    A_array = training[:num_A_train]
    B_array = training[1+num_A_train:num_A_train+num_B_train]
    C_array = training[1+num_A_train+num_B_train:]
    #Find the centroid by summing the columns and dividing by the total number of training data points in the given class
    A_centroid = A_array.mean(axis=0)
    B_centroid = B_array.mean(axis=0)
    C_centroid = C_array.mean(axis=0)
    #Calculate the weight
    AB_w = A_centroid - B_centroid
    BC_w = B_centroid - C_centroid
    AC_w = A_centroid - C_centroid
    #Calculate t
    AB_t = np.dot(AB_w, (A_centroid + B_centroid) / 2)
    BC_t = np.dot(BC_w, (B_centroid + C_centroid) / 2)
    AC_t = np.dot(AC_w, (A_centroid + C_centroid) / 2)
    #find the size of the testing data for each class
    num_A_test = testing_input[0][1]
    num_B_test = testing_input[0][2]
    num_C_test = testing_input[0][3]
    #remove the information and separate into three numpy arrays for each class
    testing_input.remove(testing_input[0])
    testing = np.array(testing_input)
    A_test_array = testing[:num_A_test]
    B_test_array = testing[num_A_test:num_A_test+num_B_test]
    C_test_array = testing[num_A_test+num_B_test:]

    truePositiveA = 0;
    truePositiveB = 0;
    truePositiveC = 0;
    trueNegativeA = 0;
    trueNegativeB = 0;
    trueNegativeC = 0;
    AinB = 0;
    AinC = 0;
    BinA = 0;
    BinC = 0;
    CinA = 0;
    CinB = 0;
    #loop through the testing data and store the true positive and true negative results. Additionally store
    #the number of A points classified as B, A points classified in C and etc.
    for i in range(num_A_test):
        if((np.dot(A_test_array[i], AB_w) >= AB_t) & (np.dot(A_test_array[i], AC_w) >= AC_t)):
            truePositiveA += 1
        elif((np.dot(A_test_array[i], AB_w) < AB_t)):
            AinB += 1
        else:
            AinC += 1
    for i in range(num_B_test):
        if((np.dot(B_test_array[i], AB_w) < AB_t) & (np.dot(B_test_array[i], BC_w) >= BC_t)):
            truePositiveB += 1
        elif((np.dot(B_test_array[i], AB_w) < AB_t)):
            BinA += 1
        else:
            BinC += 1
    for i in range(num_C_test):
        if((np.dot(C_test_array[i], AC_w) < AC_t) & (np.dot(C_test_array[i], BC_w) < BC_t)):
            truePositiveC += 1
        elif((np.dot(C_test_array[i], AC_w) < AC_t)):
            CinA += 1
        else:
            CinB += 1
    #Calculate the true positive, true negative, false positive, false negative, total positive, total negative
    #and estimated positive to calculate the tpr, fpr, error rate, accuracy and precision
    truePositive = truePositiveA + truePositiveB + truePositiveC
    trueNegative = truePositiveB + truePositiveC + BinC + CinB + truePositiveA + truePositiveB + AinB + BinA +truePositiveA + truePositiveC + AinC + CinA
    falsePositive = BinA + CinA + AinB + CinB + AinC + BinC
    falseNegative = AinC + AinB + BinA + BinC + CinA + CinB
    totalPositive = truePositive + falseNegative
    totalNegative = falsePositive + trueNegative
    estimatedPositive = truePositive + falsePositive
    #Calculate these measures and return the result values
    return {
                "tpr": float(truePositive)/totalPositive,
                "fpr": float(falsePositive)/totalNegative,
                "error_rate": float(falsePositive+falseNegative)/(totalPositive+totalNegative),
                "accuracy": float(truePositive+trueNegative)/(totalPositive+totalNegative),
                "precision": float(truePositive)/estimatedPositive
           }
