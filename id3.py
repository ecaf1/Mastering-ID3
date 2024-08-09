import pandas as pd
from math import log2
from collections import Counter
from icecream import ic

class ID3:
    def __init__(self):
        self.X = None
        self.y = None
        self.attributes = None
        self.tree = None

    def _entropy(self, y):
        total = len(y)
        counts = Counter(y)
        entropy = -sum(
            (count / total) * log2(count / total) for count in counts.values()
        )
        return entropy

    def _rest(self, X, y, attribute):
        total = len(y)
        counts = Counter(X[attribute])
        rest = 0
        for value, count in counts.items():
            rest += (count / total) * self._entropy(y[X[attribute] == value])
        return rest

    def _info_gain(self, X, y, attribute):
        return self._entropy(y) - self._rest(X, y, attribute)

    def _best_atribute(self, X, y, attributes):
        gains = {
            attribute: self._info_gain(X, y, attribute) for attribute in attributes
        }
        return max(gains, key=gains.get)

    def _build_tree(self, X, y, attributes):
        # Base cases
        if len(Counter(y)) == 1:
            return y.iloc[0]
        if len(attributes) == 0 or X.empty:
            return y.mode()[0]
        best_attr = self._best_atribute(X, y, attributes)
        tree = {best_attr: {}}

        for value in X[best_attr].unique():
            ic(X[best_attr])
            subset_X = X[X[best_attr] == value]
            subset_y = y[X[best_attr] == value]

            if subset_y.empty:
                tree[best_attr][value] = y.mode()[0]
            else:
                subtree = self._build_tree(
                    subset_X.drop(columns=[best_attr]), subset_y, attributes - {best_attr}
                )
                tree[best_attr][value] = subtree
                
        return tree
    def fit(self, X, y):
        attributes = set(X.columns)
        self.tree = self._build_tree(X, y, attributes)

    def _predict_intance(self, instance, tree):
        if not instance(tree, dict):
            return tree
        attribute = next(iter(tree))
        value = instance[attribute]
        if value in tree[attribute]:
            return self._predict_intance(instance, tree[attribute][value])
        else:
            return None
    
    def predict(self, X):
        return X.apply(lambda instance: self._predict_intance(instance, self.tree), axis=1)
        
data = {
    "Outlook": [
        "Sunny",
        "Sunny",
        "Overcast",
        "Rain",
        "Rain",
        "Rain",
        "Overcast",
        "Sunny",
        "Sunny",
        "Rain",
        "Sunny",
        "Overcast",
        "Overcast",
        "Rain",
    ],
    "Temperature": [
        "Hot",
        "Hot",
        "Hot",
        "Mild",
        "Cool",
        "Cool",
        "Cool",
        "Mild",
        "Cool",
        "Mild",
        "Mild",
        "Mild",
        "Hot",
        "Mild",
    ],
    "Humidity": [
        "High",
        "High",
        "High",
        "High",
        "Normal",
        "Normal",
        "Normal",
        "High",
        "Normal",
        "Normal",
        "Normal",
        "High",
        "Normal",
        "High",
    ],
    "Wind": [
        "Weak",
        "Strong",
        "Weak",
        "Weak",
        "Weak",
        "Strong",
        "Strong",
        "Weak",
        "Weak",
        "Weak",
        "Strong",
        "Strong",
        "Weak",
        "Strong",
    ],
    "PlayTennis": [
        "No",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "No",
    ],
}
df = pd.DataFrame(data)
X = df.drop(columns=["PlayTennis"])
y = df["PlayTennis"]

id3 = ID3()
id3.fit(X, y)

print("Árvore de Decisão:")
print(id3.tree)

# gain_outlook = id3._info_gain(X, y, "Outlook")
# gain_temperature = id3._info_gain(X, y, "Temperature")
# gain_humidity = id3._info_gain(X, y, "Humidity")
# gain_wind = id3._info_gain(X, y, "Wind")

# print("Information Gain for 'Outlook':", gain_outlook)
# print("Information Gain for 'Temperature':", gain_temperature)
# print("Information Gain for 'Humidity':", gain_humidity)
# print("Information Gain for 'Wind':", gain_wind)
