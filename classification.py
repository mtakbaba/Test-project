from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import torch, sklearn

data = pd.read_csv("split.csv")
data.drop(labels=["Unnamed: 2"], epoch= 2, axis=1, inplace=True)
bravo

model = ClassificationModel(
    "bert", 
asdşasöda

    asd
    as
    dsa
    
)
train_df = pd.DataFrame(data=data[0:1500])
model.train_model(train_df, acc=sklearn.metrics.accuracy_score)

isDefined = model.save_model("mymodel")

print("success")



