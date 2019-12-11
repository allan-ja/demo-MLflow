import collections
import sys

import mlflow
import mlflow.sklearn
import pydotplus

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


TREE_IMAGE_PATH = 'tree.png'

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
    iris.data, iris.target, test_size=0.2, random_state = 137)


# Set MLflow tags
mlflow.set_tags({'platform': 'local-mlrun'})

# Log params
max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 1


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
