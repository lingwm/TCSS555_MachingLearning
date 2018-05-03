# course: TCSS555
# Homework 2
# date: 04/04/2018
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd


class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children.
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level=0):
        if self.children == {}:  # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)

    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}:  # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


# calculate the entropy of the attribute
def entropy(examples, target, attribute):
    dataEntropy = 0.0
    counts_rows_all = examples[attribute].count()
    grouped = examples.groupby(attribute)
    counts_rows_per_value = grouped.size()
    portions = []
    for i in counts_rows_per_value:
        portions.append(i/counts_rows_all)

    entropies = []
    singleEntropy = 0.0
    for group in grouped.groups:
        sub_df = grouped.get_group(group)
        grouped_by_target = sub_df.groupby(target)
        probs = []
        for i in grouped_by_target.size():
            probs.append(i/sub_df[target].count())
        for p in probs:
            singleEntropy += -p*math.log2(p)
        entropies.append(singleEntropy)
        singleEntropy = 0.0

    k = 0
    while k < len(portions):
        dataEntropy += portions[k]*entropies[k]
        k = k + 1
    return dataEntropy


def id3_implement(examples, target, attributes, labels):
    # If all examples are positive, Return the single-node tree Root, with label = positive
    # If all examples are negative, Return the single-node tree Root, with label = negative
    if labels[0] not in examples[target].tolist():
        return DecisionNode(labels[1])
    if labels[1] not in examples[target].tolist():
        return DecisionNode(labels[0])

    # If attributes is empty, Return the single-node tree Root, with label = most common value of target in examples
    if not attributes:
        counts = examples[target].value_counts()
        return DecisionNode(counts.keys()[0])

    # chose the best classifier attribute
    entropys = dict()
    for attribute in attributes:
        entropys[attribute] = entropy(examples, target, attribute)
    attribute_classify = min(entropys, key=entropys.get)
    attributes.remove(attribute_classify)

    # create the decision root
    root = DecisionNode(attribute_classify)

    # For each possible value of the classifying attribute
    grouped = examples.groupby(attribute_classify)
    for group in grouped.groups:
        sub_examples = grouped.get_group(group).drop(columns=[attribute_classify])
        root.children[group] = id3_implement(sub_examples, target, attributes, labels)
    return root


def id3(examples, target, attributes):
    # get the types of values in target column("yes, no" or "1, 0")
    labels = examples[target].unique()
    root = id3_implement(examples, target, attributes, labels)
    return root


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train, target, attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0, len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i, target]):
        correct += 1
print("\nThe accuracy is: ", correct / len(test))