import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


# to easier build the decision tree
class Node:

    def __init__(self, leaf, rows, column, feature, conclusion, value):
        self.leaf = leaf
        self.rows = rows
        self.column = column
        self.feature = feature
        self.conclusion = conclusion
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class DecisionTree:

    def __init__(self):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.root = None
        self.super_x = None
        self.nodes = []
        self.rules = []

    def set_super_X(self, X):
        self.super_x = X

    def split(self, X, y):
        column_names = X.head()
        root_entropy = entropy(np.array([len([i for i in y[X.index] if i == y.unique()[
                               0]]), len([i for i in y[X.index] if i == y.unique()[1]])]))
        left_node = []
        nodes = []

        best_super_partition = 0
        best_feature = ""
        other_features = []
        best_column = ""

        filtered_columns = [
            item for item in column_names if item not in self.rules]

        for column in filtered_columns:
            super_entropy = 0
            possible_feature = []
            partition = [0, 0]

            # Find possible features
            for row in X.loc[:, column]:
                if row not in possible_feature:
                    possible_feature.append(row)

            for feature in possible_feature:
                partition = [0, 0]
                partition_l = X.loc[X[column] == feature]
                sub_div = y.unique()

                for i in partition_l.index:
                    if y[i] == sub_div[0]:
                        partition[0] += 1
                    else:
                        partition[1] += 1

                super_entropy += entropy(np.array(partition)) * \
                    len(partition_l)/len(X)

            if root_entropy - super_entropy > best_super_partition:
                best_super_partition = root_entropy - super_entropy
                best_feature = feature
                best_column = column
                left_node = partition_l.index
                other_features = [s for s in possible_feature if s != feature]

        nodes = [X.loc[X[best_column] ==
                       feature].index for feature in other_features]
        all_nodes = [left_node] + nodes

        all_features = [best_feature] + other_features

        return best_column, best_feature, all_features, all_nodes

    def fit(self, X, y, depth=0):

        residual_table = X

        remaining_integers = np.array(residual_table.index.tolist())

        # determine tree division from root
        new_comp = self.split(residual_table, y)

        # best_partition, best_question, best_feature, left_node, right_node
        best_question = new_comp[0]
        best_feature = new_comp[1]
        other_features = new_comp[2]
        all_nodes = new_comp[3]

        # lets start by initializing the root node
        parent = None

        if depth == 0:
            root_node = Node(False, residual_table.index.tolist(),
                             best_question, best_feature, None, "Root")
            self.nodes.append(root_node)
            parent = root_node
            self.set_super_X(X)
        else:
            parent = self.nodes[-1]

        for i in range(len(all_nodes)):
            if len(all_nodes[i]) == 0:
                return

            if self.is_leaf(y, all_nodes[i], depth):
                node = Node(True, all_nodes[i],
                            best_question, best_feature, self.conclusion(y, all_nodes[i], depth), other_features[i])

                self.nodes.append(node)
                parent.add_child(node)

                remaining_integers = [
                    x for x in remaining_integers if x not in all_nodes[i]]
                continue

            node = Node(False, all_nodes[i],
                        best_question, best_feature, self.conclusion(y, all_nodes[i], depth), other_features[i])

            self.nodes.append(node)
            parent.add_child(node)

            remaining_integers = [
                x for x in remaining_integers if x not in all_nodes[i]]

            self.fit(self.super_x.iloc[all_nodes[i]], y, depth + 1)

        # self.print_tree(self.nodes[0])

        return

    def print_tree(self, node, depth=0):
        if node:
            print("       " * depth + str(depth) + " " +
                  str(node.column) + " " + str(node.value) + " " + str(node.conclusion))
            for child in node.children:
                self.print_tree(child, depth + 1)

    def traverse_tree(self, node, features):
        if node:
            for child in node.children:

                if child.value == features[child.column]:
                    if child.leaf:

                        return child.conclusion

                    result = self.traverse_tree(child, features)
                    if result is not None:
                        return result

            return node.conclusion

    def predict(self, X):
   
        predictions = []

        current_node = self.nodes[0]
        for index, row in X.iterrows():
            predictions.append(self.traverse_tree(current_node, row))

        return np.array(predictions)

    def get_rules(self):


        decision_tree = self.extract(self.nodes[0])

        return decision_tree

    def extract(self, node):

        if not node:
            return []

        if node.leaf:

            return [([(node.column, node.value)], node.conclusion)]

        decisions_and_conclusions = []
        for child in node.children:
            sub_results = self.extract(child)
            for result in sub_results:
                if node.value != "Root":
                    result[0].insert(0, (node.column, node.value))
            decisions_and_conclusions.extend(sub_results)

        return decisions_and_conclusions

    def is_leaf(self, y, indecies, depth):
        leaf = False

        sub_div = y.unique()

        partition = [0, 0]

        for i in indecies:

            if y[i] == sub_div[0]:
                partition[0] += 1
            else:
                partition[1] += 1

        if entropy(np.array(partition)) == 0 or depth == 4:
            leaf = True
        return leaf

    def conclusion(self, y, indecies, depth):
        alternatives = y.unique()
        a = 0
        b = 0
        for i in indecies:
            if y[i] == alternatives[0]:
                a += 1
            else:
                b += 1
        if a > b:
            return alternatives[0]
        return alternatives[1]


# --- Some utility functions

def accuracy(y_true, y_pred):


    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):

    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))
