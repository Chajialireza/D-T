import argparse
from pandas.core.frame import DataFrame
import math
from numpy.ma.core import MAError
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve 
import os
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import xlsxwriter

index10=1

class Node:
     
    def __init__(self, data_idx, impurity_method, node_level, impurity_value, nfeatures):
        self.data_idx = data_idx
        self.impurity_method = impurity_method
        self.impurity_value = impurity_value
        self.node_level = node_level
        self.nfeatures = nfeatures

        # default value below
        self.dfeature = -1  # the feature based on which decision will be made
        self.majority_class = -1  # the majority class label at the given node
        self.left_child = None  # reference to the left_child
        self.right_child = None  # reference to the right_child

    # static method to initialize the node, takes the same parameters as required to create a Node
    @staticmethod
    def _init_node(data_idx, impurity_method, level, impurity_value, nfeatures):
        return Node(data_idx, impurity_method, level, impurity_value, nfeatures)

    # static method to get the counts of each label in a given labels_list
    @staticmethod
    def get_label_counts(actual_labels_list):
        # getting all the labels from the training set
        labels = set(train_y)

        # list to store the counts of each label
        count_of_labels = []
        for label in labels:
            # getting the count of each label from the list
            count = len(actual_labels_list[actual_labels_list == label])
            if count != 0:
                count_of_labels.append(count)
        return count_of_labels

    # static method to get the weighted impurity
    @staticmethod
    def get_weighted_impurity(impurity_left, impurity_right, count_left, count_right):
        total_sum = count_left + count_right
        return impurity_left * (count_left / total_sum) + impurity_right * (count_right / total_sum)

    # static method to get the indices based on the feature which equals to expected value
    # We are getting the indices from the training data for the respective index and comparing with the
    # expected feature value
    @staticmethod
    def get_indices_for_feature(indices, feature_num, expected_value):
        # creating a set to store the results
        final_index_set = []
        for index in indices:
            # getting the feature value from the train data for the particular index and feature number
            feature_value = train_x[index, feature_num]
            #  if the feature value equals the expected value adding it to the list
            if feature_value == expected_value:
                final_index_set.append(index)
        return final_index_set

    # method to build the decision tree
    # returns the root of the decision tree
    @staticmethod
    def buildDT(_data, _indices, _impurity_method, _nl, _p_threshold):
        
        _initial_impurity_value = -1
        # calculating the initial gini index at the root
        if _impurity_method == "gini":
            _initial_impurity_value = Node.calculateGINI(_indices)
        elif _impurity_method == "entropy":
            _initial_impurity_value = Node.calculateEntropy(_indices)
        elif _impurity_method == "t-entropy":
            _initial_impurity_value = Node.calculateTEntropy(_indices)
        elif _impurity_method == "tsallis":
            _initial_impurity_value = Node.calculateTsallis(_indices)
        elif _impurity_method == "renyi":
            _initial_impurity_value = Node.calculateRenyi(_indices)
        else:
            raise ("Invalid impurity method provided " + str(_impurity_method))

        # creating the root of the decision tree using _init_node method
        col_=np.shape(_data)[1]
        #print(col_)
        _decision_tree = Node._init_node(_indices,
                                         impurity_method=_impurity_method,
                                         level=0,
                                         impurity_value=_initial_impurity_value,
                                         nfeatures=range(_data.shape[1])
                                         #nfeatures=range(col_)
                                         )
                
        # splitting the root node until the max level or impurity threshold is reached
        _decision_tree.split_node(max_levels=_nl, _p_threshold=_p_threshold)
        return _decision_tree

    # calculates the gini index for the given set of indices
    @staticmethod
    def calculateGINI(indices):
        # returning 0 when length of indices is 0, does not matter which value return because weighted
        # index will make sure that this values is ignored
        if len(indices) == 0:
            return 0

        # getting the labels of the indices provided
        final_indexed_labels = train_y[indices]

        # getting the count of each label for the given indices
        count_of_each_labels = Node.get_label_counts(final_indexed_labels)

        # converting to np array for easier processing
        counts = np.array(count_of_each_labels)

        # getting the probability of each label count
        probabilities = np.divide(counts, np.sum(counts))

        # squaring the probabilities and summing them
        sum_sqaure_probabilities = np.sum(np.power(probabilities, 2))
        return 1 - sum_sqaure_probabilities
    

    @staticmethod
    def calculateTsallis(indices):
        # returning 0 when length of indices is 0, does not matter which value return because weighted
        # index will make sure that this values is ignored
        if len(indices) == 0:
            return 0

        # getting the labels of the indices provided
        final_indexed_labels = train_y[indices]

        # getting the count of each label for the given indices
        count_of_each_labels = Node.get_label_counts(final_indexed_labels)

        # converting to np array for easier processing
        counts = np.array(count_of_each_labels)

        # getting the probability of each label count
        probabilities = np.divide(counts, np.sum(counts))

        # squaring the probabilities and summing them
        sum_sqaure_probabilities = np.sum(np.power(probabilities, index10))
        index_=index10-1
        return (1 - sum_sqaure_probabilities)*(1/index_)

    # calculates the entropy for the given set of indices
    @staticmethod
    def calculateEntropy(indices):
        # returning 0 when length of indices is 0, does not matter which value return because weighted
        # index will make sure that this values is ignored
        if len(indices) == 0:
            return 0
        # getting the labels of the indices provided
        final_indexed_labels = train_y[indices]
        # getting the count of each label for the given indices
        count_of_each_labels = Node.get_label_counts(final_indexed_labels)
        counts = np.array(count_of_each_labels)
        # getting the probability of each label count
        probabilities = np.divide(counts, np.sum(counts))
        # calculating entropy
        entropy = np.sum(-probabilities * (np.log(probabilities) / np.log(2)))
        return entropy

    @staticmethod
    
    def calculateTEntropy(indices):
        # returning 0 when length of indices is 0, does not matter which value return because weighted
        # index will make sure that this values is ignored
        if len(indices) == 0:
            return 0
        # getting the labels of the indices provided
        final_indexed_labels = train_y[indices]

        # getting the count of each label for the given indices
        count_of_each_labels = Node.get_label_counts(final_indexed_labels)
        counts = np.array(count_of_each_labels)

        # getting the probability of each label count
        probabilities1 = np.divide(counts, np.sum(counts))
        # calculating t-entropy
        p11=np.power(probabilities1,index10)
        tentropy = np.sum(probabilities1 * (np.arctan(1/p11) ))-((np.pi)/4)
        return tentropy

    @staticmethod
    def calculateRenyi(indices):
        # returning 0 when length of indices is 0, does not matter which value return because weighted
        # index will make sure that this values is ignored
        if len(indices) == 0:
            return 0
        # getting the labels of the indices provided
        final_indexed_labels = train_y[indices]

        # getting the count of each label for the given indices
        count_of_each_labels = Node.get_label_counts(final_indexed_labels)
        counts = np.array(count_of_each_labels)

        # getting the probability of each label count
        probabilities = np.divide(counts, np.sum(counts))

        # calculating Renyi-entropy
        index_=1-index10
        renyi1 =np.log2(np.sum(np.power(probabilities,index10)))

        return (1/index_)*renyi1

    # utility method to get the majority class label in a given set of indices
    @staticmethod
    def get_maximum_occurring_label(indices):
        labels = set(train_y)
        actual_labels_list = train_y[indices]
        max_occurring_label = -1
        max_count = -1
        for label in labels:
            count = len(actual_labels_list[actual_labels_list == label])
            if count > max_count:
                max_count = count
                max_occurring_label = label

        return max_occurring_label

    # method to split a given node with maximum depth of tree could be max_levels
    # and with a impurity threshold of _p_threshold
    def split_node(self, max_levels, _p_threshold):
        # assigning the majority class at the given node, helpful in debugging
        # and we will need it for classification
        self.majority_class = Node.get_maximum_occurring_label(self.data_idx)

        # checking if the node is reached the required limits to stop the splitting,
        # it is an early stopping mechanism
        if self.node_level < max_levels and self.impurity_value > _p_threshold:

            # setting the default values which will be calculated in the next steps
            max_gain = -1
            split_feature = -1
            final_left_indices = []
            final_right_indices = []
            final_left_impurity = -1
            final_right_impurity = -1

            # Looping through each feature in the feature set provided
            for feature in self.nfeatures:
                # get the indices for the left child node if the feature value is 0
                left_indices = self.get_indices_for_feature(self.data_idx, feature, 0)
                # get the indices for the right child node if the feature value is 1
                right_indices = self.get_indices_for_feature(self.data_idx, feature, 1)
                # calculate impurity for left child and right child
                p_left = self.calculate_ip(left_indices)
                p_right = self.calculate_ip(right_indices)

                total_sum = len(left_indices) + len(right_indices)
                # calculate the weighted impurity
                if(total_sum!=0):
                    m = p_left * (len(left_indices) / total_sum) + p_right * (len(right_indices) / total_sum)
                else: m=0

                # calculate the gain
                gain = self.impurity_value - m

                # if gain is greater than max_gain update the below values
                if gain > max_gain:
                    split_feature = feature
                    max_gain = gain
                    final_left_indices = left_indices
                    final_right_indices = right_indices
                    final_left_impurity = p_left
                    final_right_impurity = p_right

            # once the above is loop is completed assign the feature value
            # through which the node can make a decision during inference.
            # note that the we get this value based on the max_gain
            self.dfeature = split_feature

            # create a node if length of the left child indices is greater than 0, else left child will be null
            # Also, note that the final_left_indices are calculated based on the
            # feature which provides the max gain and we get this value from the first loop where
            # we loop through all the features to get the max gain
            if len(final_left_indices) > 0:
                # initialize the node

                self.left_child = self._init_node(final_left_indices,
                                                  impurity_method=self.impurity_method,
                                                  level=self.node_level + 1,
                                                  impurity_value=final_left_impurity,
                                                  nfeatures=self.nfeatures)
                # split the left child
                self.left_child.split_node(max_levels, p_threshold)

            # create a node if length of the right child indices is greater than 0,
            # else right child will be null
            if len(final_right_indices) > 0:
                self.right_child = self._init_node(final_right_indices,
                                                   impurity_method=self.impurity_method,
                                                   level=self.node_level + 1,
                                                   impurity_value=final_right_impurity,
                                                   nfeatures=self.nfeatures)

                self.right_child.split_node(max_levels, p_threshold)

    # calculating impurities based on gini or entropy, if any other option is provided it will raise an error
    def calculate_ip(self, indices):
        if self.impurity_method == "gini":
            return self.calculateGINI(indices)
        elif self.impurity_method == "entropy":
            return self.calculateEntropy(indices)
        elif self.impurity_method == "t-entropy":
            return self.calculateTEntropy(indices)
        elif self.impurity_method == "tsallis":
            return self.calculateTsallis(indices)
        elif self.impurity_method == "renyi":
            return self.calculateRenyi(indices)
        else:
            raise ("Invalid impurity method provided: " + str(self.impurity_method))

    # classify method to predict the labels of the test data
    # takes two arguments test data and an output file where it appends the output label
    def classify(self, _test_x, _output_file):
        # opens the output file
        _file = open(_output_file, mode="w", encoding="utf-8")

        # looping through all the records in test data
        for record in _test_x:
            # getting the predicted label and writes in the file
            _predicted_label = self.get_predicted_label(record)
            _file.write(str(_predicted_label))
            _file.write("\n")
        _file.close()

    # method to get the prediction of the record, recursively finds out the prediction
    def get_predicted_label(self, _test_x_record):
        # if both the left child and right child are null of the current node
        # then we have reached the leaf node
        # and we will make the decision on the current nodes majority class
        if self.right_child is None and self.left_child is None:
            return self.majority_class
        else:
            # get the feature value based on the feature the node is split
            distinguishing_feature_value = _test_x_record[self.dfeature]
            # go to left child if feature value is 0 else go to right child
            if distinguishing_feature_value == 0:
                return self.left_child.get_predicted_label(_test_x_record)
            else:
                return self.right_child.get_predicted_label(_test_x_record)


# load the data file and label file using numpy library
def load_data_file_and_label(data_file, label_file):
    data = np.genfromtxt(data_file).astype(int)
    label = np.genfromtxt(label_file).astype(int)
    return data, label


# creating a arg parser for the main method to take the input arguments
def get_parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-data')
    _parser.add_argument('-nlevels')
    _parser.add_argument('-pthrd')
    _parser.add_argument('-impurity')
    _parser.add_argument('-pred_file')
    
    return _parser


# calculates accuracy based on the predicted labels and expected labels
def get_accuracy(expected, predicted):
    count = 0
    for i in range(0, len(expected)):
        if expected[i] == predicted[i]:
            count = count + 1
    return count / len(expected)

def get_precision_recall(expected, predicted):
    #global precision
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for j in np.arange(0, expected.size):
        # positive true condition
        if expected[j] == 1:
            # true positive condition
            if 1 == predicted[j]:
                true_positive += 1

            else:
                false_negative += 1
        # negative true condition
        else:
            # false positive condition
            if 1 == predicted[j]:
                false_positive += 1
            else:
                true_negative += 1
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        percision=-100
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall=-100
        
    try:
        specificity=true_negative/(false_positive+true_negative)
    except ZeroDivisionError:
        specificity=-100
        
    try:
        #f1_score=2*(recall*precision)/float(recall+ precision)
        f1_score=2*true_positive/(2*true_positive+false_negative+false_positive)
    except ZeroDivisionError:
        f1_score=-100
        
    try:
        NPV=true_negative/(true_negative+false_negative)
    except ZeroDivisionError:
        NPV=-100
        
    FPR=1-specificity
    FNR=1-recall
    FDR=1-precision
    MAError=mean_absolute_error(expected, predicted)
    MSError=mean_squared_error(expected, predicted)
    return precision, recall, specificity,f1_score,FPR,FNR,NPV,FDR, MAError, MSError



# main method where the program starts
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    data_file = os.path.abspath(str(args.data))
    head, tail = os.path.split(data_file)
    head1, tail1 = os.path.split(head)
    lines = []
    newList=[]
    with open(data_file) as f:
        lines = f.readlines()
        col1=0
        for i in lines:
            if(tail=="SPECT.txt"): newList.append(i.split(',')) 
            elif(tail=="TTT.txt"): newList.append(i.split(',')) 
            elif(tail=="QAR.csv"): newList.append(i.split(';'))
            elif(tail=="QOT.csv"): newList.append(i.split(';')) 
            elif(tail=="Balance.txt"): newList.append(i.split(','))
            elif(tail=="Heart.csv"):  newList.append(i.split(','))
            elif(tail=="Skin.txt"):  newList.append(i.split('   '))
    dataset = pd.DataFrame(newList)
    count_row = len(dataset)
    cr=count_row -1
    
    #################### SPECT dataset
    if(tail=="SPECT.txt"):
        dataset.replace('0\n', 0, inplace=True)
        dataset.replace('0', 0, inplace=True)
        dataset.replace('1\n', 1, inplace=True)
        count_col=len(dataset.columns)-1
        X = dataset.drop(dataset.columns[count_col], axis=1) 
        Y = dataset[dataset.columns[count_col]]
        for i0 in range(len(X)):
            for j0 in range(count_col):
                X.iloc[i0][j0]=int(X.iloc[i0][j0])
    
    train_x1, test_x1, train_y1, test_y1 =train_test_split(X, Y, test_size = 0.25,random_state=0)
    train_x=train_x1.to_numpy()
    train_y=train_y1.to_numpy()
    test_x=test_x1.to_numpy()
    test_y=test_y1.to_numpy()
    #print(DataFrame.head(Y, n=25))
   
    # get all the arguments required for the program to run
    max_level_input = int(args.nlevels)
    p_threshold = float(args.pthrd)
    impurity = str(args.impurity)
    pred_output_file = os.path.abspath(str(args.pred_file))
    row0= len(train_x)
    print(impurity)
    # build decision tree
    if(impurity=="entropy" or impurity=="gini"):
        decision_tree = Node.buildDT(train_x,
                                    _indices=list(range(row0)), 
                                    _impurity_method=impurity,
                                    _nl=max_level_input,
                                    _p_threshold=p_threshold)

        # classify the test data
        model=decision_tree.classify(test_x, _output_file=pred_output_file)
        # get the predictions from the output file to calculate the accuracy
        predictions = np.genfromtxt(pred_output_file).astype(int)
        _accuracy = get_accuracy(test_y, predictions)
        _precision_recall = get_precision_recall(test_y, predictions)
        #print(confusion_matrix(test_y, predictions))
        #print (classification_report(test_y, predictions))

        #precision, recall, specificity,f1_score,FPR,FNR,NPV,FDR, MAError, MSError
        print("Accuracy is: " + str(_accuracy))
        print("Precision is: " + str(_precision_recall[0]))
        print("Recall is: " + str(_precision_recall[1]))
        print("Specificity is: "+ str(_precision_recall[2]))
        print("F1 Score is: "+ str(_precision_recall[3]))
        print ("FPR is: "+ str(_precision_recall[4]))
        print ("FNR is: "+ str(_precision_recall[5]))
        print ("NPV is: "+ str(_precision_recall[6]))
        print ("FDR is: "+ str(_precision_recall[7]))
        print ("MAE is: "+ str(_precision_recall[8]))
        print ("MSE is: "+ str(_precision_recall[9]))
        fpr, tpr, thresholds = roc_curve(test_y, predictions)
        roc_auc = roc_auc_score(test_y, predictions)
        print("ROC_AUC is: " + str(roc_auc))
        
    elif(impurity=="t-entropy"):
        workbook = xlsxwriter.Workbook(head1+'/output/tentropy_values.xlsx')
        worksheet = workbook.add_worksheet("result")
        worksheet.write(0, 0,  "T")
        worksheet.write(0, 1,  "Sensitivity")
        worksheet.write(0, 2,  "Specificity")
        worksheet.write(0, 3,  "Precision")
        worksheet.write(0, 4,  "Accuracy")
        worksheet.write(0, 5,  "F1 Score")
        worksheet.write(0, 6,  "NPV")
        worksheet.write(0, 7,  "FDR")
        worksheet.write(0, 8,  "FPR")
        worksheet.write(0, 9,  "FNR")
        worksheet.write(0, 10,  "MAE")
        worksheet.write(0, 11,  "MSE")
        ma=1
        arr_list=np.arange(1, 301, 1)
        for index15 in range(len(arr_list)):
            try:
                index10=arr_list[index15]
                decision_tree = Node.buildDT(train_x,
                                        _indices=list(range(row0)), 
                                        _impurity_method=impurity,
                                        _nl=max_level_input,
                                        _p_threshold=p_threshold)

            # classify the test data
                decision_tree.classify(test_x, _output_file=pred_output_file)

            # get the predictions from the output file to calculate the accuracy
                predictions = np.genfromtxt(pred_output_file).astype(int)

                _accuracy = get_accuracy(test_y, predictions)
                #print(test_y,"**********",predictions)
                _precision_recall = get_precision_recall(test_y, predictions)
                print(index10)
                    
                print("Accuracy is: " + str(_accuracy))
                print("Precision is: " + str(_precision_recall[0]))
                print("Recall is: " + str(_precision_recall[1]))
                print("Specificity is: "+ str(_precision_recall[2]))
                print("F1 Score is: "+ str(_precision_recall[3]))
                print ("FPR is: "+ str(_precision_recall[4]))
                print ("FNR is: "+ str(_precision_recall[5]))
                print ("NPV is: "+ str(_precision_recall[6]))
                print ("FDR is: "+ str(_precision_recall[7]))
                print ("MAE is: "+ str(_precision_recall[8]))
                print ("MSE is: "+ str(_precision_recall[9]))
                fpr, tpr, thresholds = roc_curve(test_y, predictions)
                roc_auc = roc_auc_score(test_y, predictions)
                print("ROC_AUC is: " + str(roc_auc))
                    
                worksheet.write(ma, 0, index10 )
                worksheet.write(ma, 1,  str(round(_precision_recall[1],4)))
                worksheet.write(ma, 2,  str(round(_precision_recall[2],4)))
                worksheet.write(ma, 3,  str(round(_precision_recall[0],4)))
                worksheet.write(ma, 4,  str(round(_accuracy,4)))
                worksheet.write(ma, 5,  str(round(_precision_recall[3],4)))
                worksheet.write(ma, 6,  str(round(_precision_recall[6],4)))
                worksheet.write(ma, 7,  str(round(_precision_recall[7],4)))
                worksheet.write(ma, 8,  str(round(_precision_recall[4],4)))
                worksheet.write(ma, 9,  str(round(_precision_recall[5],4)))
                worksheet.write(ma, 10,  str(round(_precision_recall[8],4)))
                worksheet.write(ma, 11,  str(round(_precision_recall[9],4)))
                ma=ma+1
            except NameError:
                continue
        workbook.close()
    elif(impurity=="tsallis"):
        workbook = xlsxwriter.Workbook(head1+'/output/tsallisentropy_values.xlsx')
        worksheet = workbook.add_worksheet("result")
        worksheet.write(0, 0,  "Q")
        worksheet.write(0, 1,  "Sensitivity")
        worksheet.write(0, 2,  "Specificity")
        worksheet.write(0, 3,  "Precision")
        worksheet.write(0, 4,  "Accuracy")
        worksheet.write(0, 5,  "F1 Score")
        worksheet.write(0, 6,  "NPV")
        worksheet.write(0, 7,  "FDR")
        worksheet.write(0, 8,  "FPR")
        worksheet.write(0, 9,  "FNR")
        worksheet.write(0, 10,  "MAE")
        worksheet.write(0, 11,  "MSE")
        ma=1
        arr_list=np.arange(2, 301, 1)
        for index15 in range(len(arr_list)):
            try:
                index10=arr_list[index15]
                decision_tree = Node.buildDT(train_x,
                                        _indices=list(range(row0)), 
                                        _impurity_method=impurity,
                                        _nl=max_level_input,
                                        _p_threshold=p_threshold)

            # classify the test data
                decision_tree.classify(test_x, _output_file=pred_output_file)

            # get the predictions from the output file to calculate the accuracy
                predictions = np.genfromtxt(pred_output_file).astype(int)

                _accuracy = get_accuracy(test_y, predictions)
                #print(test_y,"**********",predictions)
                _precision_recall = get_precision_recall(test_y, predictions)
                print(index10)
                print("Accuracy is: " + str(_accuracy))
                print("Precision is: " + str(_precision_recall[0]))
                print("Recall is: " + str(_precision_recall[1]))
                print("Specificity is: "+ str(_precision_recall[2]))
                print("F1 Score is: "+ str(_precision_recall[3]))
                print ("FPR is: "+ str(_precision_recall[4]))
                print ("FNR is: "+ str(_precision_recall[5]))
                print ("NPV is: "+ str(_precision_recall[6]))
                print ("FDR is: "+ str(_precision_recall[7]))
                print ("MAE is: "+ str(_precision_recall[8]))
                print ("MSE is: "+ str(_precision_recall[9]))
                fpr, tpr, thresholds = roc_curve(test_y, predictions)
                roc_auc = roc_auc_score(test_y, predictions)
                print("ROC_AUC is: " + str(roc_auc))
               
                worksheet.write(ma, 0, index10 )
                worksheet.write(ma, 1,  str(round(_precision_recall[1],4)))
                worksheet.write(ma, 2,  str(round(_precision_recall[2],4)))
                worksheet.write(ma, 3,  str(round(_precision_recall[0],4)))
                worksheet.write(ma, 4,  str(round(_accuracy,4)))
                worksheet.write(ma, 5,  str(round(_precision_recall[3],4)))
                worksheet.write(ma, 6,  str(round(_precision_recall[6],4)))
                worksheet.write(ma, 7,  str(round(_precision_recall[7],4)))
                worksheet.write(ma, 8,  str(round(_precision_recall[4],4)))
                worksheet.write(ma, 9,  str(round(_precision_recall[5],4)))
                worksheet.write(ma, 10,  str(round(_precision_recall[8],4)))
                worksheet.write(ma, 11,  str(round(_precision_recall[9],4)))
                ma=ma+1
            except NameError:
                continue
        workbook.close()
    elif(impurity=="renyi"):
        workbook = xlsxwriter.Workbook(head1+'/output/renyientropy_values.xlsx')
        worksheet = workbook.add_worksheet("result")
        worksheet.write(0, 0,  "a")
        worksheet.write(0, 1,  "Sensitivity")
        worksheet.write(0, 2,  "Specificity")
        worksheet.write(0, 3,  "Precision")
        worksheet.write(0, 4,  "Accuracy")
        worksheet.write(0, 5,  "F1 Score")
        worksheet.write(0, 6,  "NPV")
        worksheet.write(0, 7,  "FDR")
        worksheet.write(0, 8,  "FPR")
        worksheet.write(0, 9,  "FNR")
        worksheet.write(0, 10,  "MAE")
        worksheet.write(0, 11,  "MSE")
        ma=1
        arr_list=np.arange(2, 301, 1)
        for index15 in range(len(arr_list)):
            try:
                index10=arr_list[index15]
                decision_tree = Node.buildDT(train_x,
                                        _indices=list(range(row0)), 
                                        _impurity_method=impurity,
                                        _nl=max_level_input,
                                        _p_threshold=p_threshold)

            # classify the test data
                decision_tree.classify(test_x, _output_file=pred_output_file)

            # get the predictions from the output file to calculate the accuracy
                predictions = np.genfromtxt(pred_output_file).astype(int)

                _accuracy = get_accuracy(test_y, predictions)
                #print(test_y,"**********",predictions)
                _precision_recall = get_precision_recall(test_y, predictions)
                print(index10)
                
                print("Accuracy is: " + str(_accuracy))
                print("Precision is: " + str(_precision_recall[0]))
                print("Recall is: " + str(_precision_recall[1]))
                print("Specificity is: "+ str(_precision_recall[2]))
                print("F1 Score is: "+ str(_precision_recall[3]))
                print ("FPR is: "+ str(_precision_recall[4]))
                print ("FNR is: "+ str(_precision_recall[5]))
                print ("NPV is: "+ str(_precision_recall[6]))
                print ("FDR is: "+ str(_precision_recall[7]))
                print ("MAE is: "+ str(_precision_recall[8]))
                print ("MSE is: "+ str(_precision_recall[9]))
                fpr, tpr, thresholds = roc_curve(test_y, predictions)
                roc_auc = roc_auc_score(test_y, predictions)
                print("ROC_AUC is: " + str(roc_auc))
                   
                worksheet.write(ma, 0, index10 )
                worksheet.write(ma, 1,  str(round(_precision_recall[1],4)))
                worksheet.write(ma, 2,  str(round(_precision_recall[2],4)))
                worksheet.write(ma, 3,  str(round(_precision_recall[0],4)))
                worksheet.write(ma, 4,  str(round(_accuracy,4)))
                worksheet.write(ma, 5,  str(round(_precision_recall[3],4)))
                worksheet.write(ma, 6,  str(round(_precision_recall[6],4)))
                worksheet.write(ma, 7,  str(round(_precision_recall[7],4)))
                worksheet.write(ma, 8,  str(round(_precision_recall[4],4)))
                worksheet.write(ma, 9,  str(round(_precision_recall[5],4)))
                worksheet.write(ma, 10,  str(round(_precision_recall[8],4)))
                worksheet.write(ma, 11,  str(round(_precision_recall[9],4)))
                ma=ma+1
            except NameError:
                continue
        workbook.close()


    
   
    


