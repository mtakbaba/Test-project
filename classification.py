from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import torch, sklearn

data = pd.read_csv("split.csv")
data.drop(labels=["Unnamed: 2"], axis=1, inplace=True)

X = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model_args = {
    "use_early_stopping": True,
    "early_stopping_delta": 0.01,
    "early_stopping_metric": "mcc",
    "early_stopping_metric_minimize": False,
    "early_stopping_patience": 5,
    "evaluate_during_training_steps": 1000,
    "fp16": False,
    "num_train_epochs":3
}

model = ClassificationModel(
    "bert", 
    "dbmdz/bert-base-turkish-cased",
    use_cuda=False,
    args=model_args, 
    num_labels=10
)
train_df = pd.DataFrame(data=data[0:1500])
model.train_model(train_df, acc=sklearn.metrics.accuracy_score)

model.save_model("mymodel")




