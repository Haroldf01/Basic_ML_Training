import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
target = iris.target

print(features)
print(target)

decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(features, target)

observation = [[ 5, 2, 3, 2]] # Predict observation's class
print(model.predict(observation))
print(model.predict_proba(observation))

# from sklearn import tree
# from IPython.display import Image

# dot_data = tree.export_graphviz(
#     decisiontree, out_file=None,
#     feature_names=iris.feature_names,
#     class_names=iris.target_names
# )
# graph = pydotplus.graph_from_dot_data(dot_data) # Show graph
# Image(graph.create_png())

# references

# - https://towardsdatascience.com/decision-trees-60707f06e836
# - https://blog.paperspace.com/decision-trees/
