# coding: utf-8

from math import log

data_set = [['a', 1, 'yes'],
            ['a', 1, 'yes'],
            ['a', 0, 'no'],
            ['b', 1, 'no'],
            ['b', 1, 'no']]


class decisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def uniquecounts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)
    ent = 0.0
    for r in results:
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def divide_set(rows, column, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def get_column_values(rows, col):
    column_values = {}

    for row in rows:
        column_values[row[col]] = 1
    return column_values


def target_score(p, set1, set2, scoref=entropy):
    return p * scoref(set1) + (1 - p) * scoref(set2)


def gain(rows, scoref=entropy):
    current_score = scoref(rows)
    print current_score
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1

    for col in range(0, column_count):
        column_values = get_column_values(rows, col)

        for value in column_values.keys():
            (set1, set2) = divide_set(rows, col, value)

            # gain = beforeSplit() – afterSplit()
            p = float(len(set1)) / len(rows)
            gain = current_score - target_score(p, set1, set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    return (best_gain, best_criteria, best_sets)


print gain(data_set)


def buildtree(rows, scoref=entropy):
    if len(rows) == 0: return decisionNode()
    current_score = scoref(rows)

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # Generate the list of different values in
        # this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # Now try dividing the rows up for each value
        # in this column
        for value in column_values.keys():
            (set1, set2) = divide_set(rows, col, value)

            # Information gain
            p = float(len(set1)) / len(rows)

            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # Create the sub branches
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return decisionNode(results=uniquecounts(rows))


def print_tree(tree, indent=''):
    if tree.results != None:
        print str(tree.results)
    else:
        print str(tree.col) + ':' + str(tree.value) + '?'

        print indent + 'T->', print_tree(tree.tb, indent + ' ')
        print indent + 'F->', print_tree(tree.fb, indent + ' ')


def classify(observation, tree):
    if tree.results != None:  # is not a leafnode
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


tree = buildtree(data_set)
print tree.col
print tree.value
print tree.results
print tree.fb.results
print_tree(tree)

print classify(['a', 1], tree)


def prune(tree, mingain):
    # 分支不是叶节点 进行剪纸
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)

    if tree.tb.results != None and tree.fb.results != None:
        tb, fb = [], []
        print tree.tb.results.items()
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)

        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)



# prune(tree, 0.5)
# print_tree(tree)
