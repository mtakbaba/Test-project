from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import torch, sklearn

data = pd.read_csv("split.csv")
data.drop(labels=["Unnamed: 2"], axis=1, inplace=True)
bravo

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


dfjghdfjgüdfü
fogkfjgdsf
ewıgurwvıjdsf
dfkgjdfjhger
vdfjbdfgd
djgjfgdfgdfg
fkjgdfg
4gfıgjkdfg
3rgdlkgdf
g4gkjfgkdfg
34gkdfjg
rdfgdsfsdf
ldfkjgnkwef
dlkjgekg
gkjgbkjdsfkgjdsf
rgkdsjfgkds
dogfjnkedfrgjnsdgf
dfkgbjdfjfgkdjsg
dslgjdskgfjdsfg
dsfgkdsjfgkdsgsafg
felnkgnjkljsdfg
sdfkgjdskfgd
fgdjfghdjfghdfg
dfgldkfgd
fgdfkgdjfgkjdfg
dfgldkfgld
gdfgjkdfgjkdsjfgd
fgdkfjgdgf
dfkjgnksfg
sdfgdkjsfgkdjfg
dfgldkfgd
fgdsfgjdfgd
fgdsfgkjdfg
dsfgdsfgdsfgswdfsdfgv
desrgsdfgsdfg
sdrfsdfgewrfv
sdfkjsngfsdfglsdjfngk
gdsfjgndskfjge
rgsdlfjglkajgrqaf
gsdflgjskldfg
eqrgbljdsnfbkadjfgk
gvsdlfjvakdf
fgsdjfgdsafgasdf
dfljgnsdlfg
sdfgkmdsflkgsdfgldsjfngksdfg
dfogujndskfjgq
wergsdüfgbkjqrnegpoıdgf
e
grpısjdfglkeqrg
sdfogbpkjdptglker
vdspşfıjperüg
adfb*sdjogıjeqrvd
