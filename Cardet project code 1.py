#!/usr/bin/env python
# coding: utf-8

# pip install -U openpyxl

# In[2]:


import sys
print(sys.executable)


# In[3]:


import openpyxl
print(openpyxl.__file__)
print(openpyxl.__version__)


# In[5]:


import pandas as pd

xls = pd.ExcelFile("ProW_Reporting data T1.xlsx")

print(xls.sheet_names)


# In[6]:


df = pd.read_excel(
    "ProW_Reporting data T1.xlsx",
    sheet_name=0
)

print(df.columns.tolist())


# In[8]:


child = pd.read_excel(
    "Student_datasets_recoded.xlsx",
    sheet_name="T1_Children"
)
print(child.columns.tolist())


# In[9]:


print(teacher.shape)
student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)

print(student.shape)
print(student.columns.tolist()[:50])


# In[7]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

# 1. LOAD DATA

teacher = pd.read_excel(
    "ProW_Reporting data T1.xlsx",
    sheet_name="T1_Teachers"
)

child = pd.read_excel(
    "Student_datasets_recoded.xlsx",
    sheet_name="T1_Children"
)

# 2. TEACHER FEATURES

prefixes = [
    "TSWQ_",
    "PERMA_",
    "TSES-short_",
    "TSSES_",
    "JobSat_",
    "MBI_",
    "PCS_"
]

teacher_features = []

for col in teacher.columns:

    for p in prefixes:

        if col.startswith(p):

            teacher_features.append(col)

print("Teacher features:", len(teacher_features))

# 3. CLEAN


X_teacher = teacher[teacher_features].copy()

X_teacher = X_teacher.replace(
    [999, 99, 98, 97],
    np.nan
)

imp = SimpleImputer(strategy="median")

X_teacher = imp.fit_transform(X_teacher)

scaler = StandardScaler()

X_teacher = scaler.fit_transform(X_teacher)

# 4. TEACHER EMBEDDING

pca = PCA(
    n_components=10,
    random_state=42
)

embedding = pca.fit_transform(X_teacher)

embed_df = pd.DataFrame(
    embedding,
    columns=[f"E{i}" for i in range(1,11)]
)

embed_df["Teacher_ID"] = teacher["Teacher_ID"]

# 5. CHILD OUTCOME

sdq_cols = [
    c
    for c in child.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

print("SDQ items:", len(sdq_cols))

child["SDQ_TOTAL"] = (
    child[sdq_cols]
    .replace([999,99,98,97], np.nan)
    .sum(axis=1)
)

# 6. MERGE

data = child.merge(
    embed_df,
    left_on="Teacher ID",
    right_on="Teacher_ID",
    how="inner"
)

print("Merged N =", len(data))

# 7. MODEL

X = data[
    [f"E{i}" for i in range(1,11)]
]

y = data["SDQ_TOTAL"]

rf = RandomForestRegressor(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    rf,
    X,
    y,
    scoring="r2",
    cv=cv
)


print("Mean R2")
print(scores.mean())


print(scores)

# 8. FIT FINAL MODEL

rf.fit(X,y)

importance = pd.DataFrame({
    "Embedding":[f"E{i}" for i in range(1,11)],
    "Importance":rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Embeddings")
print(importance.head(10))


# In[12]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

from umap import UMAP
import hdbscan

import matplotlib.pyplot as plt


# LOAD DATA


student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)


# ITEM SELECTION


prefixes = [
    "SDQ_",
    "ASBI_",
    "CBRS_",
    "ECBC_"
]

item_cols = []

for c in student.columns:

    if "_rec" in c:

        for p in prefixes:

            if c.startswith(p):

                item_cols.append(c)

print("Items:", len(item_cols))


# PREPROCESS


X = student[item_cols].copy()

X = X.replace(
    [999,99,98,97],
    np.nan
)

imp = SimpleImputer(strategy="median")

X = imp.fit_transform(X)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)


# AUTOENCODER


class AutoEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(X.shape[1],64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,X.shape[1])
        )

    def forward(self,x):

        z = self.encoder(x)

        xhat = self.decoder(z)

        return xhat

model = AutoEncoder()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.MSELoss()


# TRAIN


for epoch in range(500):

    optimizer.zero_grad()

    xhat = model(X_tensor)

    loss = loss_fn(
        xhat,
        X_tensor
    )

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:

        print(epoch, loss.item())


# EMBEDDING


with torch.no_grad():

    embedding = (
        model.encoder(X_tensor)
        .numpy()
    )

print(embedding.shape)


# UMAP

umap_model = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

coords = umap_model.fit_transform(
    embedding
)

# HDBSCAN

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30
)

clusters = clusterer.fit_predict(
    embedding
)

# VISUALIZE

plt.figure(figsize=(8,6))

plt.scatter(
    coords[:,0],
    coords[:,1],
    c=clusters
)

plt.title(
    "Child Behavioral Phenotypes"
)

plt.show()

# SAVE

student["Phenotype"] = clusters

for i in range(16):

    student[f"Embed_{i+1}"] = embedding[:,i]

student.to_csv(
    "child_embedding_results.csv",
    index=False
)

print(
    student["Phenotype"]
    .value_counts()
)


# In[13]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(n_components=10)

embedding = pca.fit_transform(X)

km = KMeans(
    n_clusters=4,
    random_state=42
)

cluster = km.fit_predict(
    embedding
)

print(
    pd.Series(cluster)
    .value_counts()
)


# In[14]:


student["cluster"] = cluster

sdq_cols = [c for c in student.columns
            if c.startswith("SDQ_")
            and c.endswith("_rec")]

student["SDQ_TOTAL"] = student[sdq_cols].sum(axis=1)

student.groupby("cluster")["SDQ_TOTAL"].agg(
    ["count","mean","std"]
)


# In[15]:


asbi_cols = [c for c in student.columns
             if c.startswith("ASBI_")
             and c.endswith("_rec")]

student["ASBI_TOTAL"] = student[asbi_cols].sum(axis=1)

student.groupby("cluster")["ASBI_TOTAL"].agg(
    ["mean","std"]
)


# In[16]:


cbrs_cols = [c for c in student.columns
             if c.startswith("CBRS_")
             and c.endswith("_rec")]

student["CBRS_TOTAL"] = student[cbrs_cols].sum(axis=1)

student.groupby("cluster")["CBRS_TOTAL"].agg(
    ["mean","std"]
)


# In[17]:


cluster_profile = []

for c in sorted(student["cluster"].unique()):

    temp = student[
        student["cluster"] == c
    ]

    means = temp[item_cols].mean()

    cluster_profile.append(means)

cluster_profile = pd.DataFrame(
    cluster_profile
)

cluster_profile.index = sorted(
    student["cluster"].unique()
)

cluster_profile


# In[18]:


cluster_profile.T


# In[19]:


from scipy.stats import f_oneway

for col in [
    "SDQ_TOTAL",
    "ASBI_TOTAL",
    "CBRS_TOTAL"
]:
    
    groups = [
        student.loc[
            student["cluster"] == k,
            col
        ]
        for k in sorted(student["cluster"].unique())
    ]
    
    F,p = f_oneway(*groups)
    
    print(col)
    print("F =",F)
    print("p =",p)
    print()


# In[21]:


print(student.columns[-10:])


# In[23]:


print(student.columns.tolist())


# In[24]:


import pandas as pd
import numpy as np


teacher = pd.read_excel(
    "ProW_Reporting data T1.xlsx",
    sheet_name="T1_Teachers"
)

student = pd.read_csv(
    "child_embedding_results.csv"
)



required_cols = ["Teacher ID", "Phenotype"]

for c in required_cols:
    if c not in student.columns:
        raise ValueError(f"Missing column: {c}")



teacher_cluster = pd.crosstab(
    student["Teacher ID"],
    student["Phenotype"]
)

teacher_cluster_prop = (
    teacher_cluster
    .div(
        teacher_cluster.sum(axis=1),
        axis=0
    )
)

teacher_cluster_prop.columns = [
    f"Phenotype_{c}"
    for c in teacher_cluster_prop.columns
]

teacher_cluster_prop = (
    teacher_cluster_prop
    .reset_index()
)


merged = teacher_cluster_prop.merge(
    teacher,
    left_on="Teacher ID",
    right_on="Teacher_ID",
    how="inner"
)


teacher_cluster.to_csv(
    "teacher_cluster_counts.csv"
)

teacher_cluster_prop.to_csv(
    "teacher_cluster_proportions.csv",
    index=False
)

merged.to_csv(
    "teacher_cluster_merged.csv",
    index=False
)



print(
    student["Phenotype"]
    .value_counts()
    .sort_index()
)


print(
    teacher_cluster.shape[0]
)



# In[25]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)



item_cols = []

for c in student.columns:

    if c.endswith("_rec"):

        if (
            c.startswith("SDQ_")
            or c.startswith("ASBI_")
            or c.startswith("CBRS_")
        ):
            item_cols.append(c)

print("Items =", len(item_cols))



X = student[item_cols].copy()

X = X.replace(
    [999,99,98,97],
    np.nan
)

imp = SimpleImputer(
    strategy="median"
)

X = imp.fit_transform(X)

scaler = StandardScaler()

X = scaler.fit_transform(X)


pca = PCA(
    n_components=10,
    random_state=42
)

embedding = pca.fit_transform(X)


km = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

cluster = km.fit_predict(
    embedding
)

student["cluster"] = cluster

print("\nCluster counts")
print(
    pd.Series(cluster)
    .value_counts()
)



student.to_csv(
    "student_kmeans_clusters.csv",
    index=False
)

print(
    "\nSaved: student_kmeans_clusters.csv"
)


# In[26]:


import pandas as pd

teacher = pd.read_excel(
    "ProW_Reporting data T1.xlsx",
    sheet_name="T1_Teachers"
)

student = pd.read_csv(
    "student_kmeans_clusters.csv"
)


teacher_cluster = pd.crosstab(
    student["Teacher ID"],
    student["cluster"]
)

teacher_cluster_prop = (
    teacher_cluster
    .div(
        teacher_cluster.sum(axis=1),
        axis=0
    )
)

teacher_cluster_prop.columns = [
    f"Cluster_{c}"
    for c in teacher_cluster_prop.columns
]

teacher_cluster_prop = (
    teacher_cluster_prop
    .reset_index()
)



print(
    teacher_cluster_prop.head()
)


merged = teacher_cluster_prop.merge(
    teacher,
    left_on="Teacher ID",
    right_on="Teacher_ID",
    how="inner"
)


print(
    merged.shape
)



cluster_cols = [
    c
    for c in merged.columns
    if c.startswith("Cluster_")
]

teacher_cols = []

for c in merged.columns:

    if (
        c.startswith("PERMA_")
        or c.startswith("TSWQ_")
        or c.startswith("MBI_")
        or c.startswith("TSSES_")
        or c.startswith("TSES-short_")
        or c.startswith("JobSat_")
        or c.startswith("PCS_")
    ):
        teacher_cols.append(c)

corrs = []

for t in teacher_cols:

    for cl in cluster_cols:

        try:

            r = merged[t].corr(
                merged[cl]
            )

            corrs.append(
                [t,cl,r]
            )

        except:
            pass

corrs = pd.DataFrame(
    corrs,
    columns=[
        "Teacher_Item",
        "Cluster",
        "Correlation"
    ]
)

corrs["Abs"] = (
    corrs["Correlation"]
    .abs()
)

corrs = corrs.sort_values(
    "Abs",
    ascending=False
)



print(
    corrs.head(50)
)

corrs.to_csv(
    "teacher_cluster_correlations.csv",
    index=False
)

merged.to_csv(
    "teacher_cluster_merged.csv",
    index=False
)

print("\nSaved:")
print("teacher_cluster_correlations.csv")
print("teacher_cluster_merged.csv")


# In[27]:


import pandas as pd
import numpy as np

from scipy.stats import spearmanr



df = pd.read_csv(
    "teacher_cluster_merged.csv"
)


def scale_mean(df, prefix):

    cols = [
        c
        for c in df.columns
        if c.startswith(prefix)
    ]

    return df[cols].mean(axis=1)

df["PERMA"] = scale_mean(df, "PERMA_")
df["TSWQ"] = scale_mean(df, "TSWQ_")
df["TSSES"] = scale_mean(df, "TSSES_")
df["MBI"] = scale_mean(df, "MBI_")
df["PCS"] = scale_mean(df, "PCS_")
df["JOBSAT"] = scale_mean(df, "JobSat_")



scale_cols = [
    "PERMA",
    "TSWQ",
    "TSSES",
    "MBI",
    "PCS",
    "JOBSAT"
]

cluster_cols = [
    "Cluster_0",
    "Cluster_1",
    "Cluster_2",
    "Cluster_3"
]

results = []

for s in scale_cols:

    for c in cluster_cols:

        r,p = spearmanr(
            df[s],
            df[c]
        )

        results.append([
            s,
            c,
            r,
            p
        ])

results = pd.DataFrame(
    results,
    columns=[
        "Scale",
        "Cluster",
        "Spearman_r",
        "P"
    ]
)

results["AbsR"] = (
    results["Spearman_r"]
    .abs()
)

results = results.sort_values(
    "AbsR",
    ascending=False
)

print(results)

results.to_csv(
    "teacher_scale_cluster_correlations.csv",
    index=False
)



teacher_summary = df[
    scale_cols +
    cluster_cols
]

teacher_summary.to_csv(
    "teacher_summary.csv",
    index=False
)

print(
    "\nSaved:"
)
print(
    "teacher_scale_cluster_correlations.csv"
)
print(
    "teacher_summary.csv"
)


# In[28]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import RandomForestClassifier



teacher = pd.read_excel(
    "ProW_Reporting data T1.xlsx",
    sheet_name="T1_Teachers"
)

student = pd.read_csv(
    "student_kmeans_clusters.csv"
)



teacher_features = []

prefixes = [
    "TSWQ_",
    "PERMA_",
    "TSES-short_",
    "TSSES_",
    "JobSat_",
    "MBI_",
    "PCS_"
]

for c in teacher.columns:

    for p in prefixes:

        if c.startswith(p):

            teacher_features.append(c)

print(
    "Teacher items:",
    len(teacher_features)
)


teacher_child = (
    student.groupby("Teacher ID")["cluster"]
    .agg(
        lambda x: x.value_counts().index[0]
    )
    .reset_index()
)

teacher_child.columns = [
    "Teacher_ID",
    "DominantCluster"
]


data = teacher.merge(
    teacher_child,
    on="Teacher_ID",
    how="inner"
)

print(
    "Teachers merged:",
    len(data)
)

print(
    data["DominantCluster"]
    .value_counts()
)



X = data[
    teacher_features
].copy()

X = X.replace(
    [999,99,98,97],
    np.nan
)

imp = SimpleImputer(
    strategy="median"
)

X = imp.fit_transform(X)

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = data[
    "DominantCluster"
]


rf = RandomForestClassifier(
    n_estimators=2000,
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    rf,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print(scores)
print(scores.mean())



rf.fit(
    X,
    y
)

importance = pd.DataFrame({
    "Variable": teacher_features,
    "Importance": rf.feature_importances_
})

importance = (
    importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print(
    importance.head(50)
)

importance.to_csv(
    "rf_teacher_to_phenotype.csv",
    index=False
)

print(
    "\nSaved: rf_teacher_to_phenotype.csv"
)


# In[29]:


import pandas as pd
import numpy as np
import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier



teacher = pd.read_excel(
    "ProW_Reporting data T1.xlsx",
    sheet_name="T1_Teachers"
)

student = pd.read_csv(
    "student_kmeans_clusters.csv"
)


teacher_child = (
    student.groupby("Teacher ID")["cluster"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)

teacher_child.columns = [
    "Teacher_ID",
    "DominantCluster"
]


teacher_features = []

prefixes = [
    "TSWQ_",
    "PERMA_",
    "TSES-short_",
    "TSSES_",
    "JobSat_",
    "MBI_",
    "PCS_"
]

for c in teacher.columns:

    for p in prefixes:

        if c.startswith(p):

            teacher_features.append(c)


data = teacher.merge(
    teacher_child,
    on="Teacher_ID",
    how="inner"
)

X = data[teacher_features].copy()

X = X.replace(
    [999,99,98,97],
    np.nan
)

imp = SimpleImputer(
    strategy="median"
)

X = imp.fit_transform(X)

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = data["DominantCluster"]



rf = RandomForestClassifier(
    n_estimators=3000,
    random_state=42,
    n_jobs=-1
)

rf.fit(X,y)


explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X)


shap.summary_plot(
    shap_values,
    X,
    feature_names=teacher_features,
    max_display=30
)


if isinstance(shap_values, list):

    imp = np.mean(
        np.abs(
            np.concatenate(shap_values,axis=0)
        ),
        axis=0
    )

else:

    imp = np.mean(
        np.abs(shap_values),
        axis=0
    )

importance = pd.DataFrame({
    "Feature": teacher_features,
    "SHAP": imp
})

importance = (
    importance
    .sort_values(
        "SHAP",
        ascending=False
    )
)

print(
    importance.head(50)
)

importance.to_csv(
    "SHAP_teacher_phenotype.csv",
    index=False
)

print(
    "\nSaved: SHAP_teacher_phenotype.csv"
)


# In[30]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    "teacher_cluster_merged.csv"
)

# scale scores

df["PERMA"] = df[
    [c for c in df.columns if c.startswith("PERMA_")]
].mean(axis=1)

df["MBI"] = df[
    [c for c in df.columns if c.startswith("MBI_")]
].mean(axis=1)

df["PCS"] = df[
    [c for c in df.columns if c.startswith("PCS_")]
].mean(axis=1)

# correlations

corr_data = df[
    [
        "PERMA",
        "MBI",
        "PCS",
        "Cluster_0",
        "Cluster_1",
        "Cluster_2",
        "Cluster_3"
    ]
]

corr = corr_data.corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title(
    "Teacher Wellbeing vs Child Phenotype Composition"
)

plt.tight_layout()

plt.savefig(
    "Figure1_heatmap.png",
    dpi=300
)

plt.show()


# In[31]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


EMBED_DIM = 64
NHEAD = 4
NUM_LAYERS = 2
BATCH_SIZE = 128
EPOCHS = 300
LR = 1e-3
TEMPERATURE = 0.07

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE =", DEVICE)



teacher = pd.read_excel(
    "ProW_Reporting data T1.xlsx",
    sheet_name="T1_Teachers"
)

child = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)


teacher_features = []

teacher_prefixes = [
    "TSWQ_",
    "PERMA_",
    "TSES-short_",
    "TSSES_",
    "JobSat_",
    "MBI_",
    "PCS_"
]

for c in teacher.columns:

    for p in teacher_prefixes:

        if c.startswith(p):

            teacher_features.append(c)

print(
    "Teacher items:",
    len(teacher_features)
)



child_features = []

for c in child.columns:

    if c.endswith("_rec"):

        if (
            c.startswith("SDQ_")
            or c.startswith("ASBI_")
            or c.startswith("CBRS_")
        ):
            child_features.append(c)

print(
    "Child items:",
    len(child_features)
)



teacher_X = (
    teacher[
        ["Teacher_ID"] +
        teacher_features
    ]
    .copy()
)

teacher_X = teacher_X.replace(
    [999,99,98,97],
    np.nan
)

child_X = (
    child[
        ["Teacher ID"] +
        child_features
    ]
    .copy()
)

child_X = child_X.replace(
    [999,99,98,97],
    np.nan
)



imp_teacher = SimpleImputer(
    strategy="median"
)

teacher_X[
    teacher_features
] = imp_teacher.fit_transform(
    teacher_X[
        teacher_features
    ]
)

imp_child = SimpleImputer(
    strategy="median"
)

child_X[
    child_features
] = imp_child.fit_transform(
    child_X[
        child_features
    ]
)


teacher_scaler = StandardScaler()

teacher_X[
    teacher_features
] = teacher_scaler.fit_transform(
    teacher_X[
        teacher_features
    ]
)

child_scaler = StandardScaler()

child_X[
    child_features
] = child_scaler.fit_transform(
    child_X[
        child_features
    ]
)


data = child_X.merge(
    teacher_X,
    left_on="Teacher ID",
    right_on="Teacher_ID",
    how="inner"
)

print(
    "Pairs:",
    len(data)
)

teacher_matrix = data[
    teacher_features
].values

child_matrix = data[
    child_features
].values



teacher_tensor = torch.tensor(
    teacher_matrix,
    dtype=torch.float32
)

child_tensor = torch.tensor(
    child_matrix,
    dtype=torch.float32
)



class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embed_dim=64
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            1,
            embed_dim
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=NHEAD,
                batch_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_LAYERS
            )
        )

        self.fc = nn.Linear(
            embed_dim,
            embed_dim
        )

    def forward(
        self,
        x
    ):

        x = x.unsqueeze(-1)

        x = self.input_proj(x)

        x = self.transformer(x)

        x = x.mean(
            dim=1
        )

        x = self.fc(x)

        x = F.normalize(
            x,
            dim=1
        )

        return x



teacher_encoder = Encoder(
    len(teacher_features),
    EMBED_DIM
).to(DEVICE)

child_encoder = Encoder(
    len(child_features),
    EMBED_DIM
).to(DEVICE)

params = (
    list(
        teacher_encoder.parameters()
    )
    +
    list(
        child_encoder.parameters()
    )
)

optimizer = torch.optim.Adam(
    params,
    lr=LR
)



def contrastive_loss(
    teacher_emb,
    child_emb
):

    logits = (
        teacher_emb
        @
        child_emb.T
    ) / TEMPERATURE

    labels = torch.arange(
        len(logits)
    ).to(DEVICE)

    loss1 = F.cross_entropy(
        logits,
        labels
    )

    loss2 = F.cross_entropy(
        logits.T,
        labels
    )

    return (
        loss1 +
        loss2
    ) / 2



teacher_tensor = (
    teacher_tensor
    .to(DEVICE)
)

child_tensor = (
    child_tensor
    .to(DEVICE)
)

n = len(data)

for epoch in range(EPOCHS):

    idx = torch.randperm(n)

    total_loss = 0

    for i in range(
        0,
        n,
        BATCH_SIZE
    ):

        batch_idx = idx[
            i:i+BATCH_SIZE
        ]

        t = teacher_tensor[
            batch_idx
        ]

        c = child_tensor[
            batch_idx
        ]

        t_emb = (
            teacher_encoder(t)
        )

        c_emb = (
            child_encoder(c)
        )

        loss = contrastive_loss(
            t_emb,
            c_emb
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

    if epoch % 10 == 0:

        print(
            epoch,
            total_loss
        )



teacher_encoder.eval()
child_encoder.eval()

with torch.no_grad():

    teacher_emb = (
        teacher_encoder(
            teacher_tensor
        )
        .cpu()
        .numpy()
    )

    child_emb = (
        child_encoder(
            child_tensor
        )
        .cpu()
        .numpy()
    )



teacher_out = pd.DataFrame(
    teacher_emb,
    columns=[
        f"TEmbed_{i}"
        for i in range(
            EMBED_DIM
        )
    ]
)

child_out = pd.DataFrame(
    child_emb,
    columns=[
        f"CEmbed_{i}"
        for i in range(
            EMBED_DIM
        )
    ]
)

teacher_out.to_csv(
    "teacher_transformer_embedding.csv",
    index=False
)

child_out.to_csv(
    "child_transformer_embedding.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "teacher_transformer_embedding.csv"
)

print(
    "child_transformer_embedding.csv"
)


# In[ ]:




