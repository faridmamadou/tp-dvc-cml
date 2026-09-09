import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]

df = pd.read_csv("data/iris.csv")

train, test = train_test_split(
    df, test_size=params['split'], random_state=params['seed']
)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print(f"Train: {len(train)} lignes, Test: {len(test)} lignes")

