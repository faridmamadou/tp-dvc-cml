import pandas as pd
import pickle 
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

test = pd.read_csv("data/test.csv")
X = test.drop(columns=['target'])
y = test['target']

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X)

metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "f1_macro": f1_score(y, y_pred, average='macro')
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

ConfusionMatrixDisplay.from_predictions(y, y_pred)
plt.savefig("confusion_matrix.png")