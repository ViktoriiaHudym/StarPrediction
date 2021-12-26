import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)

# Import data
col_names = ['temperature', 'luminosity', 'radius', 'absolute magnitude', 'star type', 'star color', 'spectral class']
stars = pd.read_csv("6 class csv.csv", header=0, names=col_names)
stars.head()

# Normalizing data
labelencoder = LabelEncoder()
stars['star_color'] = labelencoder.fit_transform(stars['star color'])
stars['spectral_class'] = labelencoder.fit_transform(stars['spectral class'])
print(stars)

X = stars.drop(['star type','star color','spectral class'], axis=1)
y = stars['star type']

# Training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=43)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print(X_test)

y_pred = clf.predict(X_test)

accrc = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accrc}")

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

# Drawing decision tree
import graphviz
from sklearn import tree

feature_cols = ['temperature', 'luminosity', 'radius', 'absolute magnitude', 'star_color', 'spectral_class']
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_cols,
                                class_names=['0', '1', '2', '3', '4', '5'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("star")

# print('')
#
# y_pred1 = clf.predict([[2800, 0.000200, 0.1600, 16.65, 10, 5]])   # type - '0'
# y_pred2 = clf.predict([[2800, 0.000200, 0.1600, 15.00, 10, 5]])
# y_pred3 = clf.predict([[2800, 0.000200, 400.00, -5.00, 10, 5]])
# y_pred4 = clf.predict([[2800, 0.000200, 500.00, -8.00, 10, 5]])
# print(y_pred1, y_pred2, y_pred3, y_pred4)