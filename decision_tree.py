import collections
import sys

import mlflow
import mlflow.sklearn
import pydotplus

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


TREE_IMAGE_PATH = 'tree.png'
RANDOM_STATE = 51

def export_tree(model, local_image_path):
    dot_data = export_graphviz(model, out_file=None, 
                    feature_names = iris.feature_names,
                    class_names = iris.target_names,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)
    graph = pydotplus.graph_from_dot_data(dot_data)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    graph.write_png(local_image_path)


# Load Data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state = RANDOM_STATE)


with mlflow.start_run():
    mlflow.set_tags({'platform': 'local-pyfile'})
    # Log params
    max_depth = 3
    mlflow.log_param("max_depth", max_depth)

    # Model training
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log artifact
    export_tree(model, TREE_IMAGE_PATH)
    mlflow.log_artifact(TREE_IMAGE_PATH)

