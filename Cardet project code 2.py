#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

teacher = pd.read_csv(
    "teacher_transformer_embedding.csv"
)

child = pd.read_csv(
    "child_transformer_embedding.csv"
)

T = teacher.values
C = child.values

sim = cosine_similarity(
    T,
    C
)

top1 = 0
top5 = 0
top10 = 0

for i in range(len(sim)):

    rank = np.argsort(
        sim[i]
    )[::-1]

    if rank[0] == i:
        top1 += 1

    if i in rank[:5]:
        top5 += 1

    if i in rank[:10]:
        top10 += 1

n = len(sim)

print()

print("Top1 =", top1/n)

print("Top5 =", top5/n)

print("Top10 =", top10/n)


# In[2]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

for k in range(2,9):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    lab = km.fit_predict(emb)

    sil = silhouette_score(
        emb,
        lab
    )

    print(
        k,
        round(sil,4)
    )


# In[3]:


from umap import UMAP
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

cluster = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
).fit_predict(emb)

coord = UMAP(
    random_state=42
).fit_transform(
    emb
)

plt.figure(
    figsize=(8,6)
)

plt.scatter(
    coord[:,0],
    coord[:,1],
    c=cluster
)

plt.show()


# In[4]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity



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

child_features = []

for c in child.columns:

    if c.endswith("_rec"):

        if (
            c.startswith("SDQ_")
            or c.startswith("ASBI_")
            or c.startswith("CBRS_")
        ):
            child_features.append(c)



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


imp = SimpleImputer(
    strategy="median"
)

teacher_X[
    teacher_features
] = imp.fit_transform(
    teacher_X[
        teacher_features
    ]
)

child_X[
    child_features
] = imp.fit_transform(
    child_X[
        child_features
    ]
)



scaler = StandardScaler()

teacher_X[
    teacher_features
] = scaler.fit_transform(
    teacher_X[
        teacher_features
    ]
)

child_X[
    child_features
] = scaler.fit_transform(
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

print("Pairs =", len(data))



teacher_pca = PCA(
    n_components=32,
    random_state=42
)

child_pca = PCA(
    n_components=32,
    random_state=42
)

T = teacher_pca.fit_transform(
    data[teacher_features]
)

C = child_pca.fit_transform(
    data[child_features]
)



sim = cosine_similarity(
    T,
    C
)

top1 = 0
top5 = 0
top10 = 0
mrr = []

for i in range(len(sim)):

    rank = np.argsort(
        sim[i]
    )[::-1]

    pos = np.where(
        rank == i
    )[0][0] + 1

    mrr.append(
        1 / pos
    )

    if rank[0] == i:

        top1 += 1

    if i in rank[:5]:

        top5 += 1

    if i in rank[:10]:

        top10 += 1

n = len(sim)


print("PCA RETRIEVAL")

print("Top1 =", top1/n)
print("Top5 =", top5/n)
print("Top10 =", top10/n)
print("MRR =", np.mean(mrr))


# In[5]:


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

print("\nKMeans on Transformer Embedding\n")

for k in range(2,11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    lab = km.fit_predict(emb)

    sil = silhouette_score(
        emb,
        lab
    )

    print(
        "k=",
        k,
        "silhouette=",
        round(sil,4)
    )

best_k = 4

cluster = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
).fit_predict(
    emb
)

out = emb.copy()

out["cluster"] = cluster

out.to_csv(
    "transformer_clusters.csv",
    index=False
)

print(
    "\nSaved transformer_clusters.csv"
)


# In[7]:


import pandas as pd
import numpy as np

from scipy.stats import f_oneway
from sklearn.cluster import KMeans



student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)



emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

print("Students =", len(student))
print("Embeddings =", len(emb))



student = student.iloc[:len(emb)].copy()



sdq_cols = [
    c for c in student.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

asbi_cols = [
    c for c in student.columns
    if c.startswith("ASBI_")
    and c.endswith("_rec")
]

cbrs_cols = [
    c for c in student.columns
    if c.startswith("CBRS_")
    and c.endswith("_rec")
]

student["SDQ_TOTAL"] = (
    student[sdq_cols]
    .replace([999,99,98,97], np.nan)
    .sum(axis=1)
)

student["ASBI_TOTAL"] = (
    student[asbi_cols]
    .replace([999,99,98,97], np.nan)
    .sum(axis=1)
)

student["CBRS_TOTAL"] = (
    student[cbrs_cols]
    .replace([999,99,98,97], np.nan)
    .sum(axis=1)
)



cluster = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
).fit_predict(
    emb
)

student["TCluster"] = cluster

print()
print(student["TCluster"].value_counts())


for outcome in [
    "SDQ_TOTAL",
    "ASBI_TOTAL",
    "CBRS_TOTAL"
]:

    groups = []

    for g in sorted(
        student["TCluster"].unique()
    ):

        groups.append(
            student.loc[
                student["TCluster"] == g,
                outcome
            ]
        )

    F, p = f_oneway(*groups)

    print()
    print(outcome)
    print("F =", F)
    print("p =", p)



print()
print(
    student
    .groupby("TCluster")[
        [
            "SDQ_TOTAL",
            "ASBI_TOTAL",
            "CBRS_TOTAL"
        ]
    ]
    .mean()
)


# In[8]:


import pandas as pd
import numpy as np

from scipy.stats import spearmanr



student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)

emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

student = student.iloc[:len(emb)].copy()



sdq_cols = [
    c for c in student.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

asbi_cols = [
    c for c in student.columns
    if c.startswith("ASBI_")
    and c.endswith("_rec")
]

cbrs_cols = [
    c for c in student.columns
    if c.startswith("CBRS_")
    and c.endswith("_rec")
]

student["SDQ_TOTAL"] = student[sdq_cols].sum(axis=1)

student["ASBI_TOTAL"] = student[asbi_cols].sum(axis=1)

student["CBRS_TOTAL"] = student[cbrs_cols].sum(axis=1)



results = []

for outcome in [
    "SDQ_TOTAL",
    "ASBI_TOTAL",
    "CBRS_TOTAL"
]:

    for col in emb.columns:

        r,p = spearmanr(
            emb[col],
            student[outcome]
        )

        results.append([
            outcome,
            col,
            r,
            p,
            abs(r)
        ])

results = pd.DataFrame(
    results,
    columns=[
        "Outcome",
        "Embedding",
        "R",
        "P",
        "AbsR"
    ]
)

results = results.sort_values(
    "AbsR",
    ascending=False
)

print(
    results.head(50)
)

results.to_csv(
    "transformer_outcome_correlations.csv",
    index=False
)

print(
    "\nSaved: transformer_outcome_correlations.csv"
)


# In[9]:


import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)

emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

student = student.iloc[:len(emb)].copy()



sdq_cols = [
    c for c in student.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

student["SDQ_TOTAL"] = (
    student[sdq_cols]
    .sum(axis=1)
)



X = emb

y = student["SDQ_TOTAL"]

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = Ridge()

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print()
print("SDQ R2")
print(scores)
print(scores.mean())


# In[13]:


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
EPOCHS = 160

LR = 1e-3

TEMPERATURE = 0.07

ALPHA = 1.0      # contrastive
BETA = 0.5       # SDQ supervision

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


sdq_cols = [
    c
    for c in child.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

child["SDQ_TOTAL"] = (
    child[sdq_cols]
    .replace(
        [999,99,98,97],
        np.nan
    )
    .sum(axis=1)
)



teacher_X = (
    teacher[
        ["Teacher_ID"]
        +
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
        ["Teacher ID"]
        +
        child_features
        +
        ["SDQ_TOTAL"]
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

sdq_vector = data[
    "SDQ_TOTAL"
].values


teacher_tensor = torch.tensor(
    teacher_matrix,
    dtype=torch.float32
)

child_tensor = torch.tensor(
    child_matrix,
    dtype=torch.float32
)

sdq_tensor = torch.tensor(
    sdq_vector,
    dtype=torch.float32
)

class Encoder(nn.Module):

    def __init__(
        self,
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



class OutcomeHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                EMBED_DIM,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(
        self,
        x
    ):

        return self.net(x)



teacher_encoder = Encoder(
    EMBED_DIM
).to(DEVICE)

child_encoder = Encoder(
    EMBED_DIM
).to(DEVICE)

outcome_head = (
    OutcomeHead()
    .to(DEVICE)
)


params = (

    list(
        teacher_encoder.parameters()
    )

    +

    list(
        child_encoder.parameters()
    )

    +

    list(
        outcome_head.parameters()
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
        loss1 + loss2
    ) / 2



teacher_tensor = (
    teacher_tensor
    .to(DEVICE)
)

child_tensor = (
    child_tensor
    .to(DEVICE)
)

sdq_tensor = (
    sdq_tensor
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

        y = sdq_tensor[
            batch_idx
        ]

        t_emb = teacher_encoder(
            t
        )

        c_emb = child_encoder(
            c
        )

        pred = outcome_head(
            c_emb
        ).squeeze()

        loss_alignment = (
            contrastive_loss(
                t_emb,
                c_emb
            )
        )

        loss_sdq = (
            F.mse_loss(
                pred,
                y
            )
        )

        loss = (
            ALPHA * loss_alignment
            +
            BETA * loss_sdq
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
            round(total_loss,4)
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



pd.DataFrame(
    teacher_emb,
    columns=[
        f"TEmbed_{i}"
        for i in range(
            EMBED_DIM
        )
    ]
).to_csv(
    "teacher_multitask_transformer.csv",
    index=False
)

pd.DataFrame(
    child_emb,
    columns=[
        f"CEmbed_{i}"
        for i in range(
            EMBED_DIM
        )
    ]
).to_csv(
    "child_multitask_transformer.csv",
    index=False
)

print()
print(
    "Saved:"
)

print(
    "teacher_multitask_transformer.csv"
)

print(
    "child_multitask_transformer.csv"
)


# In[14]:


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

from scipy.stats import f_oneway


emb = pd.read_csv(
    "child_multitask_transformer.csv"
)

student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)

print("Embeddings =", len(emb))
print("Students =", len(student))



student = student.iloc[:len(emb)].copy()



sdq_cols = [
    c
    for c in student.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

student["SDQ_TOTAL"] = (
    student[sdq_cols]
    .replace([999,99,98,97], np.nan)
    .sum(axis=1)
)



asbi_cols = [
    c
    for c in student.columns
    if c.startswith("ASBI_")
    and c.endswith("_rec")
]

student["ASBI_TOTAL"] = (
    student[asbi_cols]
    .replace([999,99,98,97], np.nan)
    .sum(axis=1)
)



cbrs_cols = [
    c
    for c in student.columns
    if c.startswith("CBRS_")
    and c.endswith("_rec")
]

student["CBRS_TOTAL"] = (
    student[cbrs_cols]
    .replace([999,99,98,97], np.nan)
    .sum(axis=1)
)


X = emb.values


print()
print("LINEAR PROBE")


ridge = Ridge(alpha=1)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    ridge,
    X,
    student["SDQ_TOTAL"],
    scoring="r2",
    cv=cv
)

print()
print("SDQ R2")
print(scores)
print(scores.mean())


print()
print("KMEANS")


for k in range(2,11):

    cluster = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    ).fit_predict(X)

    sil = silhouette_score(
        X,
        cluster
    )

    print(
        "k=",
        k,
        "silhouette=",
        round(sil,4)
    )



cluster = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
).fit_predict(X)

student["MTCluster"] = cluster

print()
print(student["MTCluster"].value_counts())



print()
print("ANOVA")


for outcome in [
    "SDQ_TOTAL",
    "ASBI_TOTAL",
    "CBRS_TOTAL"
]:

    groups = []

    for g in sorted(
        student["MTCluster"].unique()
    ):

        groups.append(
            student.loc[
                student["MTCluster"]==g,
                outcome
            ]
        )

    F,p = f_oneway(*groups)

    print()
    print(outcome)
    print("F =",F)
    print("p =",p)


print()
print("CLUSTER MEANS")


print(

    student
    .groupby("MTCluster")[
        [
            "SDQ_TOTAL",
            "ASBI_TOTAL",
            "CBRS_TOTAL"
        ]
    ]
    .mean()

)


student.to_csv(
    "multitask_transformer_clusters.csv",
    index=False
)

print()
print("Saved:")
print("multitask_transformer_clusters.csv")


# In[15]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



teacher = pd.read_csv(
    "teacher_multitask_transformer.csv"
)

child = pd.read_csv(
    "child_multitask_transformer.csv"
)

teacher = teacher.values
child = child.values

print("Teacher =", len(teacher))
print("Child =", len(child))



sim = cosine_similarity(
    teacher,
    child
)


top1 = 0
top5 = 0
top10 = 0

rr = []

n = len(sim)

for i in range(n):

    ranking = np.argsort(
        sim[i]
    )[::-1]

    rank = (
        np.where(
            ranking == i
        )[0][0]
        + 1
    )

    rr.append(
        1/rank
    )

    if rank <= 1:
        top1 += 1

    if rank <= 5:
        top5 += 1

    if rank <= 10:
        top10 += 1

top1 /= n
top5 /= n
top10 /= n

mrr = np.mean(rr)

print()
print("MULTITASK RETRIEVAL")
print()

print("Top1 =",top1)
print("Top5 =",top5)
print("Top10 =",top10)
print("MRR =",mrr)


# In[16]:


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

pca.fit(X)

var_each = (
    pca.explained_variance_ratio_
)

var_total = (
    var_each.sum()
)

print()
print("PCA VARIANCE")


for i,v in enumerate(var_each):

    print(
        f"PC{i+1}: {v:.4f}"
    )

print()
print(
    "TOTAL =",
    round(
        var_total*100,
        2
    ),
    "%"
)


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["figure.dpi"] = 600

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

items = []

for c in df.columns:

    if c.endswith("_rec"):

        if (
            c.startswith("SDQ_")
            or c.startswith("ASBI_")
            or c.startswith("CBRS_")
        ):
            items.append(c)

X = df[items]

pca = PCA(
    n_components=2,
    random_state=42
)

coords = pca.fit_transform(X)

plt.figure(figsize=(8,7))

scatter = plt.scatter(
    coords[:,0],
    coords[:,1],
    c=df["cluster"],
    s=20,
    alpha=.8
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Behavioral Phenotypes"
)

plt.tight_layout()

plt.savefig(
    "Figure1_PCA_Phenotypes.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 600

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

items = []

for c in df.columns:

    if c.endswith("_rec"):

        if (
            c.startswith("SDQ_")
            or c.startswith("ASBI_")
            or c.startswith("CBRS_")
        ):
            items.append(c)

heat = (
    df
    .groupby("cluster")[items]
    .mean()
)

plt.figure(
    figsize=(18,8)
)

plt.imshow(
    heat.T,
    aspect="auto"
)

plt.colorbar()

plt.yticks(
    range(len(items)),
    items,
    fontsize=6
)

plt.xticks(
    range(4),
    [
        "Cluster0",
        "Cluster1",
        "Cluster2",
        "Cluster3"
    ]
)

plt.title(
    "Behavioral Phenotype Heatmap"
)

plt.tight_layout()

plt.savefig(
    "Figure2_Phenotype_Heatmap.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 600

models = [
    "PCA",
    "Contrastive\nTransformer",
    "Multi-task\nTransformer"
]

top1 = [
    0.13,
    7.27,
    0.53
]

top5 = [
    1.19,
    33.03,
    2.51
]

top10 = [
    1.98,
    56.14,
    4.36
]

x = np.arange(
    len(models)
)

w = .25

plt.figure(
    figsize=(9,6)
)

plt.bar(
    x-w,
    top1,
    width=w,
    label="Top1"
)

plt.bar(
    x,
    top5,
    width=w,
    label="Top5"
)

plt.bar(
    x+w,
    top10,
    width=w,
    label="Top10"
)

plt.xticks(
    x,
    models
)

plt.ylabel(
    "Retrieval Accuracy (%)"
)

plt.title(
    "Teacher-Child Retrieval Performance"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure3_Retrieval.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# In[20]:


import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 600

models = [
    "PCA",
    "Contrastive",
    "Multi-task"
]

alignment = [
    1.98,
    56.14,
    4.36
]

behavior = [
    834,
    4.53,
    16.93
]

plt.figure(
    figsize=(8,6)
)

plt.scatter(
    alignment,
    behavior,
    s=300
)

for i,m in enumerate(models):

    plt.text(
        alignment[i],
        behavior[i],
        m,
        fontsize=12
    )

plt.xlabel(
    "Teacher-Child Alignment (Top10)"
)

plt.ylabel(
    "Behavior Structure (F-statistic)"
)

plt.title(
    "Alignment–Phenotype Trade-off"
)

plt.tight_layout()

plt.savefig(
    "Figure4_Tradeoff.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams.update({
    "font.size":14,
    "axes.labelsize":16,
    "axes.titlesize":18,
    "legend.fontsize":12
})

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

items = [
    c for c in df.columns
    if c.endswith("_rec")
]

X = df[items]

pca = PCA(
    n_components=2,
    random_state=42
)

coords = pca.fit_transform(X)

plt.figure(
    figsize=(8,7)
)

for cl in sorted(df["cluster"].unique()):

    idx = df["cluster"]==cl

    plt.scatter(
        coords[idx,0],
        coords[idx,1],
        s=25,
        alpha=.75,
        label=f"Cluster {cl}"
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
)

plt.title(
    "Behavioral Phenotypes in PCA Latent Space"
)

plt.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "Figure1_PCA_Phenotypes.tiff",
    dpi=1200
)

plt.show()


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":12
})

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

items = [
    c for c in df.columns
    if c.endswith("_rec")
]

heat = (
    df
    .groupby("cluster")[items]
    .mean()
)

plt.figure(
    figsize=(18,10)
)

im = plt.imshow(
    heat.T,
    aspect="auto"
)

plt.colorbar(
    im,
    label="Mean Score"
)

plt.xticks(
    range(len(heat.index)),
    [f"Phenotype {i}" for i in heat.index]
)

plt.yticks(
    range(len(items)),
    items,
    fontsize=5
)

plt.title(
    "Behavioral Profiles Across Phenotypes"
)

plt.xlabel(
    "Phenotype"
)

plt.ylabel(
    "Behavioral Item"
)

plt.tight_layout()

plt.savefig(
    "Figure2_Phenotype_Heatmap.tiff",
    dpi=1200
)

plt.show()


# In[24]:


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size":14
})

models = [
    "PCA",
    "Contrastive\nTransformer",
    "Multi-task\nTransformer"
]

top1 = [0.13,7.27,0.53]
top5 = [1.19,33.03,2.51]
top10 = [1.98,56.14,4.36]

x = np.arange(
    len(models)
)

w = 0.25

fig,ax = plt.subplots(
    figsize=(9,6)
)

ax.bar(
    x-w,
    top1,
    width=w,
    label="Top-1"
)

ax.bar(
    x,
    top5,
    width=w,
    label="Top-5"
)

ax.bar(
    x+w,
    top10,
    width=w,
    label="Top-10"
)

ax.set_ylabel(
    "Retrieval Accuracy (%)"
)

ax.set_xlabel(
    "Model"
)

ax.set_title(
    "Teacher–Child Retrieval Benchmark"
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "Figure3_Retrieval_Benchmark.tiff",
    dpi=1200
)

plt.show()


# In[25]:


import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":14
})

models = [
    "PCA",
    "Contrastive",
    "Multi-task"
]

alignment = [
    1.98,
    56.14,
    4.36
]

behavior = [
    834.08,
    4.53,
    16.93
]

plt.figure(
    figsize=(8,6)
)

plt.scatter(
    alignment,
    behavior,
    s=400
)

for i,m in enumerate(models):

    plt.annotate(
        m,
        (
            alignment[i],
            behavior[i]
        ),
        xytext=(8,8),
        textcoords="offset points"
    )

plt.xlabel(
    "Teacher–Child Alignment (Top-10 Retrieval %)"
)

plt.ylabel(
    "Behavior Structure (ANOVA F Statistic)"
)

plt.title(
    "Alignment–Phenotype Trade-off"
)

plt.tight_layout()

plt.savefig(
    "Figure4_Tradeoff.tiff",
    dpi=1200
)

plt.show()


# In[27]:


import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":14
})

fig, ax = plt.subplots(
    figsize=(12,4)
)

ax.axis("off")



ax.text(
    0.05,
    0.5,
    "Teacher Items\n(n=139)",
    bbox=dict(
        boxstyle="round,pad=0.5"
    ),
    ha="center"
)



ax.text(
    0.28,
    0.5,
    "Teacher\nTransformer",
    bbox=dict(
        boxstyle="round,pad=0.5"
    ),
    ha="center"
)



ax.text(
    0.50,
    0.5,
    "Shared\nLatent Space\n(64D)",
    bbox=dict(
        boxstyle="circle,pad=0.6"
    ),
    ha="center"
)


ax.text(
    0.72,
    0.5,
    "Child\nTransformer",
    bbox=dict(
        boxstyle="round,pad=0.5"
    ),
    ha="center"
)


ax.text(
    0.95,
    0.5,
    "Child Items\n(n=82)",
    bbox=dict(
        boxstyle="round,pad=0.5"
    ),
    ha="center"
)


ax.arrow(
    0.12,
    0.5,
    0.10,
    0,
    head_width=0.03,
    length_includes_head=True
)

ax.arrow(
    0.35,
    0.5,
    0.10,
    0,
    head_width=0.03,
    length_includes_head=True
)

ax.arrow(
    0.88,
    0.5,
    -0.10,
    0,
    head_width=0.03,
    length_includes_head=True
)

ax.arrow(
    0.65,
    0.5,
    -0.10,
    0,
    head_width=0.03,
    length_includes_head=True
)

plt.title(
    "Teacher–Child Contrastive Transformer Framework",
    fontsize=18,
    pad=20
)

plt.savefig(
    "Figure5_Transformer_Framework.tiff",
    dpi=1200,
    bbox_inches="tight"
)

plt.show()


# In[29]:


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size":14,
    "axes.labelsize":16,
    "axes.titlesize":18,
    "legend.fontsize":12
})


epochs = np.array([
0,10,20,30,40,50,60,70,
80,90,100,110,120,130,
140,150
])

loss = np.array([
5061.1937,
4606.7031,
3614.7816,
2320.5676,
1170.2243,
462.0907,
172.4157,
102.6487,
88.1428,
77.7507,
70.7806,
66.4663,
63.2406,
56.4196,
53.1449,
50.1475
])


fig, ax = plt.subplots(
    figsize=(8,6)
)

ax.plot(
    epochs,
    loss,
    linewidth=3,
    marker="o",
    markersize=6,
    label="Training Loss"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Loss"
)

ax.set_title(
    "Multi-Task Teacher–Child Transformer Training Convergence"
)

ax.legend(
    frameon=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "Figure6_Training_Loss.tiff",
    dpi=1200,
    bbox_inches="tight"
)

plt.savefig(
    "Figure6_Training_Loss.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: Figure6_Training_Loss.tiff"
)


# In[30]:


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import umap

import matplotlib.pyplot as plt


emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

emb = emb.values

print(
    "Embeddings =",
    emb.shape
)



cluster = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
).fit_predict(
    emb
)



reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

coords = reducer.fit_transform(
    emb
)


plt.rcParams.update({
    "font.size":14
})

plt.figure(
    figsize=(8,7)
)

for c in np.unique(cluster):

    idx = cluster==c

    plt.scatter(
        coords[idx,0],
        coords[idx,1],
        s=30,
        alpha=.8,
        label=f"Cluster {c}"
    )

plt.title(
    "Contrastive Transformer Latent Space"
)

plt.xlabel(
    "UMAP-1"
)

plt.ylabel(
    "UMAP-2"
)

plt.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "Figure7_Contrastive_UMAP.tiff",
    dpi=1200
)

plt.show()


# In[31]:


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import umap

import matplotlib.pyplot as plt

emb = pd.read_csv(
    "child_multitask_transformer.csv"
)

emb = emb.values

cluster = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
).fit_predict(
    emb
)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

coords = reducer.fit_transform(
    emb
)

plt.rcParams.update({
    "font.size":14
})

plt.figure(
    figsize=(8,7)
)

for c in np.unique(cluster):

    idx = cluster==c

    plt.scatter(
        coords[idx,0],
        coords[idx,1],
        s=30,
        alpha=.8,
        label=f"Cluster {c}"
    )

plt.title(
    "Multi-task Transformer Latent Space"
)

plt.xlabel(
    "UMAP-1"
)

plt.ylabel(
    "UMAP-2"
)

plt.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "Figure8_Multitask_UMAP.tiff",
    dpi=1200
)

plt.show()


# In[32]:


import numpy as np



top10 = 0.5614


n = 757

scores = np.random.binomial(
    1,
    top10,
    size=(5000,n)
)

means = scores.mean(
    axis=1
)

lower = np.percentile(
    means,
    2.5
)

upper = np.percentile(
    means,
    97.5
)

print()

print(
    "Top10 =",
    round(top10,4)
)

print(
    "95% CI =",
    round(lower,4),
    "-",
    round(upper,4)
)


# In[35]:


import pandas as pd

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

print(df.columns.tolist())


# In[36]:


import pandas as pd

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

print(df.head())

print("\nColumns:\n")

for c in df.columns:
    print(c)


# In[38]:


print(df.shape)

print(df.columns[-20:])

print(df.columns.tolist()[-20:])


# In[40]:


for c in df.columns:
    if "cluster" in c.lower():
        print(c)


# In[41]:


print(
    df.groupby("Phenotype")[
        [
            "SDQ_TOTAL",
            "ASBI_TOTAL",
            "CBRS_TOTAL"
        ]
    ]
    .agg(
        ["mean","std","count"]
    )
)


# In[42]:


means = (
    df.groupby("Phenotype")[
        [
            "SDQ_TOTAL",
            "ASBI_TOTAL",
            "CBRS_TOTAL"
        ]
    ]
    .mean()
)

stds = (
    df.groupby("Phenotype")[
        [
            "SDQ_TOTAL",
            "ASBI_TOTAL",
            "CBRS_TOTAL"
        ]
    ]
    .std()
)

counts = (
    df["Phenotype"]
    .value_counts()
    .sort_index()
)

print("\nMEANS\n")
print(means)

print("\nSTDS\n")
print(stds)

print("\nCOUNTS\n")
print(counts)


# In[43]:


import pandas as pd

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

print(df.columns.tolist())

print(df["cluster"].value_counts())


# In[44]:


import pandas as pd
import numpy as np

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

student = pd.read_excel(
    "Student_datasets_recoded.xlsx"
)

student = student.iloc[:len(df)].copy()

student["cluster"] = df["cluster"]

sdq_cols = [
    c for c in student.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

asbi_cols = [
    c for c in student.columns
    if c.startswith("ASBI_")
    and c.endswith("_rec")
]

cbrs_cols = [
    c for c in student.columns
    if c.startswith("CBRS_")
    and c.endswith("_rec")
]

student["SDQ_TOTAL"] = student[sdq_cols].sum(axis=1)
student["ASBI_TOTAL"] = student[asbi_cols].sum(axis=1)
student["CBRS_TOTAL"] = student[cbrs_cols].sum(axis=1)

print(
    student.groupby("cluster")[
        [
            "SDQ_TOTAL",
            "ASBI_TOTAL",
            "CBRS_TOTAL"
        ]
    ]
    .agg(["mean","std","count"])
)


# In[45]:


import pandas as pd
import numpy as np

df = pd.read_csv(
    "student_kmeans_clusters.csv"
)

print(df.columns[-5:])


cluster_col = df.columns[-1]

print(
    "Cluster column =",
    cluster_col
)

# totals

sdq_cols = [
    c for c in df.columns
    if c.startswith("SDQ_")
    and c.endswith("_rec")
]

asbi_cols = [
    c for c in df.columns
    if c.startswith("ASBI_")
    and c.endswith("_rec")
]

cbrs_cols = [
    c for c in df.columns
    if c.startswith("CBRS_")
    and c.endswith("_rec")
]

df["SDQ_TOTAL"] = df[sdq_cols].sum(axis=1)

df["ASBI_TOTAL"] = df[asbi_cols].sum(axis=1)

df["CBRS_TOTAL"] = df[cbrs_cols].sum(axis=1)

print(
    df.groupby(cluster_col)[
        [
            "SDQ_TOTAL",
            "ASBI_TOTAL",
            "CBRS_TOTAL"
        ]
    ]
    .agg(["mean","std","count"])
)


# In[47]:


label_map = {
    0:"High Adaptive Functioning",
    1:"High Behavioral Risk",
    2:"Moderate Adaptive Functioning",
    3:"Behavioral Vulnerability"
}

for c in np.unique(cluster):

    idx = cluster == c

    plt.scatter(
        coords[idx,0],
        coords[idx,1],
        s=35,
        alpha=0.8,
        label=label_map[c]
    )


# In[48]:


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import umap
import matplotlib.pyplot as plt



emb = pd.read_csv(
    "child_transformer_embedding.csv"
)

emb = emb.values

print(
    "Embeddings =",
    emb.shape
)

cluster = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
).fit_predict(
    emb
)



label_map = {
    0:"High Adaptive Functioning",
    1:"High Behavioral Risk",
    2:"Moderate Adaptive Functioning",
    3:"Behavioral Vulnerability"
}



reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

coords = reducer.fit_transform(
    emb
)



plt.rcParams.update({
    "font.size":14
})

plt.figure(
    figsize=(9,7)
)

for c in np.unique(cluster):

    idx = cluster == c

    plt.scatter(
        coords[idx,0],
        coords[idx,1],
        s=40,
        alpha=.85,
        label=label_map[c]
    )

plt.title(
    "Behavioral Phenotypes in Contrastive Transformer Latent Space",
    fontsize=18,
    pad=15
)

plt.xlabel(
    "UMAP Dimension 1"
)

plt.ylabel(
    "UMAP Dimension 2"
)

plt.legend(
    frameon=False,
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    "Figure7_Contrastive_UMAP.tiff",
    dpi=1200,
    bbox_inches="tight"
)

plt.savefig(
    "Figure7_Contrastive_UMAP.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved Figure7_Contrastive_UMAP"
)


# In[49]:


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import umap
import matplotlib.pyplot as plt



emb = pd.read_csv(
    "child_multitask_transformer.csv"
)

emb = emb.values

print(
    "Embeddings =",
    emb.shape
)



cluster = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
).fit_predict(
    emb
)

label_map = {
    0:"High Adaptive Functioning",
    1:"High Behavioral Risk",
    2:"Moderate Adaptive Functioning",
    3:"Behavioral Vulnerability"
}



reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

coords = reducer.fit_transform(
    emb
)


plt.rcParams.update({
    "font.size":14
})

plt.figure(
    figsize=(9,7)
)

for c in np.unique(cluster):

    idx = cluster == c

    plt.scatter(
        coords[idx,0],
        coords[idx,1],
        s=40,
        alpha=.85,
        label=label_map[c]
    )

plt.title(
    "Behavioral Phenotypes in Multi-task Transformer Latent Space",
    fontsize=18,
    pad=15
)

plt.xlabel(
    "UMAP Dimension 1"
)

plt.ylabel(
    "UMAP Dimension 2"
)

plt.legend(
    frameon=False,
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    "Figure8_Multitask_UMAP.tiff",
    dpi=1200,
    bbox_inches="tight"
)

plt.savefig(
    "Figure8_Multitask_UMAP.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved Figure8_Multitask_UMAP"
)


# In[50]:


import matplotlib.pyplot as plt
import numpy as np

models = [
    "PCA",
    "Contrastive\nTransformer",
    "Multi-task\nTransformer"
]

top1 = [
    0.13,
    7.27,
    0.53
]

top5 = [
    1.19,
    33.03,
    2.51
]

top10 = [
    1.98,
    56.14,
    4.36
]

x = np.arange(
    len(models)
)

width = 0.25

plt.figure(
    figsize=(10,7)
)

plt.bar(
    x-width,
    top1,
    width,
    label="Top-1"
)

plt.bar(
    x,
    top5,
    width,
    label="Top-5"
)

plt.bar(
    x+width,
    top10,
    width,
    label="Top-10"
)

plt.ylabel(
    "Retrieval Accuracy (%)"
)

plt.title(
    "Teacher-Child Retrieval Performance Across Representation Learning Methods",
    fontsize=18,
    pad=15
)

plt.xticks(
    x,
    models
)

plt.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "Figure9_Retrieval_Benchmark.tiff",
    dpi=1200,
    bbox_inches="tight"
)

plt.savefig(
    "Figure9_Retrieval_Benchmark.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved Figure9_Retrieval_Benchmark"
)


# In[51]:


import pandas as pd

summary = pd.DataFrame({

"Metric":[

"Teacher Items",
"Child Items",
"Teachers",
"Children",
"Linked Pairs",

"PCA Variance Explained",

"Autoencoder HDBSCAN",

"PCA Cluster 0",
"PCA Cluster 1",
"PCA Cluster 2",
"PCA Cluster 3",

"SDQ ANOVA",
"ASBI ANOVA",
"CBRS ANOVA",

"PCS Cluster0",

"RF Accuracy",
"Majority Baseline",

"Contrastive Top1",
"Contrastive Top5",
"Contrastive Top10",

"Contrastive Top10 CI",

"PCA Top10",

"Multitask Top1",
"Multitask Top5",
"Multitask Top10",

"Multitask CBRS F",
"Multitask CBRS p"

],

"Value":[

139,
82,
66,
770,
757,

"64.09%",

"100% noise",

339,
97,
217,
117,

"161.79",
"682.54",
"834.08",

"r=.333 p=.008",

"60.5%",
"63.5%",

"7.27%",
"33.03%",
"56.14%",

"52.71%-59.71%",

"1.98%",

"0.53%",
"2.51%",
"4.36%",

"16.93",
"1.19e-10"

]

})

summary.to_csv(
    "FINAL_RESULTS_TABLE.csv",
    index=False
)

print(summary)

print(
    "\nSaved: FINAL_RESULTS_TABLE.csv"
)


# In[52]:


import pandas as pd

phenotypes = pd.DataFrame({

"Phenotype":[

"High Adaptive Functioning",
"High Behavioral Risk",
"Moderate Adaptive Functioning",
"Behavioral Vulnerability"

],

"N":[
339,
97,
217,
117
],

"SDQ":[
38.50,
47.31,
40.36,
43.38
],

"ASBI":[
67.51,
46.29,
60.18,
58.66
],

"CBRS":[
140.17,
82.72,
114.06,
114.23
]

})

phenotypes.to_csv(
    "FINAL_PHENOTYPES.csv",
    index=False
)

print(
    phenotypes
)

print(
    "\nSaved: FINAL_PHENOTYPES.csv"
)


# In[53]:


import os
import shutil
from datetime import datetime

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M"
)

outdir = f"PROJECT_BACKUP_{stamp}"

os.makedirs(
    outdir,
    exist_ok=True
)

extensions = [

".csv",
".xlsx",
".png",
".tiff",
".tex",
".py"

]

for file in os.listdir("."):

    if any(
        file.endswith(ext)
        for ext in extensions
    ):

        try:

            shutil.copy(
                file,
                os.path.join(
                    outdir,
                    file
                )
            )

        except:
            pass

print(
    "\nBackup saved to:"
)

print(outdir)


# In[55]:


import pandas as pd
import numpy as np

teacher = pd.read_csv(
    "teacher_transformer_embedding.csv"
).values

child = pd.read_csv(
    "child_transformer_embedding.csv"
).values

sim = teacher @ child.T

n = len(sim)

def retrieval_metrics(sim):

    top1 = 0
    top5 = 0
    top10 = 0

    for i in range(len(sim)):

        rank = np.argsort(
            sim[i]
        )[::-1]

        pos = np.where(
            rank == i
        )[0][0]

        if pos < 1:
            top1 += 1

        if pos < 5:
            top5 += 1

        if pos < 10:
            top10 += 1

    return np.array([
        top1/len(sim),
        top5/len(sim),
        top10/len(sim)
    ])

boot = []

for b in range(1000):

    idx = np.random.choice(
        n,
        n,
        replace=True
    )

    boot.append(
        retrieval_metrics(
            sim[idx][:,idx]
        )
    )

boot = np.array(boot)

for i,name in enumerate(
    ["Top1","Top5","Top10"]
):

    lo = np.percentile(
        boot[:,i],
        2.5
    )

    hi = np.percentile(
        boot[:,i],
        97.5
    )

    print(
        name,
        lo,
        hi
    )


# In[56]:


import matplotlib.pyplot as plt

models = [
    "PCA",
    "Contrastive",
    "Multi-task"
]

top10 = [
    1.98,
    56.14,
    4.36
]

plt.figure(
    figsize=(7,5)
)

plt.bar(
    models,
    top10
)

plt.ylabel(
    "Top-10 Retrieval Accuracy (%)"
)

plt.title(
    "Teacher–Child Alignment Performance"
)

plt.tight_layout()

plt.savefig(
    "Figure_Retrieval_Main.tiff",
    dpi=1200
)

plt.show()


# In[57]:


import numpy as np
import matplotlib.pyplot as plt

labels = [
    "SDQ",
    "ASBI",
    "CBRS"
]

data = {

"High Adaptive":[
38.50,
67.51,
140.17
],

"Moderate":[
40.36,
60.18,
114.06
],

"Vulnerable":[
43.38,
58.66,
114.23
],

"High Risk":[
47.31,
46.29,
82.72
]

}

angles = np.linspace(
    0,
    2*np.pi,
    len(labels),
    endpoint=False
)

angles = np.concatenate(
    [angles,[angles[0]]]
)

fig = plt.figure(
    figsize=(8,8)
)

ax = plt.subplot(
    111,
    polar=True
)

for name,vals in data.items():

    vals = vals + [vals[0]]

    ax.plot(
        angles,
        vals,
        linewidth=2,
        label=name
    )

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    labels
)

plt.legend(
    bbox_to_anchor=(1.2,1)
)

plt.savefig(
    "Figure_Phenotype_Radar.tiff",
    dpi=1200
)

plt.show()


# In[58]:


import os
import shutil
from datetime import datetime

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M"
)

backup = f"FINAL_PROJECT_{stamp}"

os.makedirs(
    backup,
    exist_ok=True
)

for f in os.listdir("."):

    if f.endswith(
        (
            ".csv",
            ".xlsx",
            ".png",
            ".tiff",
            ".tex",
            ".py"
        )
    ):

        try:

            shutil.copy(
                f,
                os.path.join(
                    backup,
                    f
                )
            )

        except:
            pass

print(
    "Saved:",
    backup
)


# In[ ]:




