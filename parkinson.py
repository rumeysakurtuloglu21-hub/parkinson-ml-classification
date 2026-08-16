from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.svm import SVC

parkinsons = fetch_ucirepo(id=174)

X = parkinsons.data.features
y = parkinsons.data.targets

print("Özellikler (X):")
print(X.head(10))

print("Hedef (y):")
print(y.head(10))

print(f"X boyutu:{X.shape}")
print(f"y boyut: {y.shape}")

print(f"X sütunları: {X.columns.tolist()}")
print(f"X veri tipleri:\n{X.dtypes}")
print(f"X eksik değerler:\n{X.isnull().sum()}")
print(f"X istatistikler:\n{X.describe()}")
print(f"y status değerler:\n{y['status'].value_counts()}")

X.columns = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]

print("Düzenlenmiş sütun isimleri:")
print(X.columns.tolist())

y["status"].value_counts().sort_index().plot(
    kind="bar",
    color=["#52067C", "#4E035D"]
)
plt.title("Parkinson Durumu Dağılımı")
plt.xlabel("Status")
plt.ylabel("Kişi Sayısı")
plt.xticks(
    [0, 1],
    ["Sağlıklı (0)", "Parkinson (1)"],
    rotation=0
)
plt.tight_layout()
plt.show()

plt.figure()
plt.boxplot(X)
plt.title("Özelliklerin Kutu Grafiği")
plt.xlabel("Özellikler")
plt.ylabel("Değerler")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


corr = X.corr()

plt.figure(figsize=(12, 8))

sbn.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Özellikler Arası Korelasyon")
plt.tight_layout()
plt.show()


df = X.copy()
df["status"] = y["status"]

status_corr = df.corr()["status"].sort_values(ascending=False)

print("Status ile korelasyonlar:")
print(status_corr)


plt.figure(figsize=(8, 6))
sbn.boxplot(
    data=df,
    x="status",
    y="spread1",
    color="#570F3D"
)

plt.title("Parkinson Durumuna Göre spread1 Dağılımı")
plt.xlabel("Durum")
plt.ylabel("spread1")
plt.xticks(
    [0, 1],
    ["Sağlıklı", "Parkinson"]
)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

sbn.boxplot(
    data=df,
    x="status",
    y="PPE",
    color="#9A0751"
)

plt.title("Parkinson Durumuna Göre PPE Dağılımı")
plt.xlabel("Durum")
plt.ylabel("PPE")
plt.xticks(
    [0, 1],
    ["Sağlıklı", "Parkinson"]
)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

sbn.boxplot(
    data=df,
    x="status",
    y="HNR",
    color="#9A0751"
)

plt.title("Parkinson Durumuna Göre HNR Dağılımı")
plt.xlabel("Durum")
plt.ylabel("HNR")

plt.xticks(
    [0, 1],
    ["Sağlıklı", "Parkinson"]
)

plt.tight_layout()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train, y_train.values.ravel())
y_pred=lr_model.predict(x_test)

print("Lojistik Regresyon Test verisi tahminleri:")
print(y_pred)

print(classification_report(y_test, y_pred))



#----------------------------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(x_train, y_train.values.ravel())

y_pred_knn = knn_model.predict(x_test)

print("KNN tahminleri:")
print(y_pred_knn)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))



k_scores = []

for k in range(1, 16):

    knn_model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        knn_model,
        x_train,
        y_train.values.ravel(),
        cv=5,
        scoring="accuracy"
    )

    mean_score = scores.mean()
    k_scores.append(mean_score)

    print(f"K={k}, Ortalama Accuracy={mean_score:.3f}")

best_k = range(1, 16)[k_scores.index(max(k_scores))]

print("En iyi K:", best_k)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 16),
    k_scores,
    color="#570F3D",
    marker="o"
)

plt.xlabel("K Değeri")
plt.ylabel("Ortalama Accuracy")
plt.title("KNN - K Değerlerine Göre 5-Fold Cross Validation")

plt.xticks(range(1, 16))
plt.grid(True)

plt.show()

final_knn = KNeighborsClassifier(n_neighbors=best_k)

final_knn.fit(
    x_train,
    y_train.values.ravel()
)

y_pred_final = final_knn.predict(x_test)

print("Final KNN Classification Report:")
print(classification_report(y_test, y_pred_final))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))


#---------------------------------------------------------------------------------------
#Karar ağaç modeli
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(x_train, y_train.values.ravel())

y_pred_dt = dt_model.predict(x_test)

print("Decision Tree Test Tahminleri:")
print(y_pred_dt)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, y_pred_dt))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

depth_values = []
dt_scores = []

for depth in range(1, 11):

    dt_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        dt_model,
        x_train,
        y_train.values.ravel(),
        cv=5,
        scoring="accuracy"
    )

    mean_score = scores.mean()

    depth_values.append(depth)
    dt_scores.append(mean_score)

    print(f"Max Depth={depth}, Ortalama Accuracy={mean_score:.3f}")

best_depth = depth_values[dt_scores.index(max(dt_scores))]

print("En iyi Max Depth:", best_depth)

final_dt = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

final_dt.fit(
    x_train,
    y_train.values.ravel()
)

y_pred_dt_final = final_dt.predict(x_test)

plt.figure(figsize=(8, 5))

plt.plot(
    depth_values,
    dt_scores,
    marker="o",
    color="#4E035D"
)

plt.xlabel("Max Depth")
plt.ylabel("Ortalama Accuracy")
plt.title("Decision Tree - Max Depth ve Cross Validation Accuracy")

plt.xticks(depth_values)
plt.grid(True)

plt.show()

print("Final Decision Tree Accuracy:")
print(accuracy_score(y_test, y_pred_dt_final))

print("\nFinal Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt_final))

print("\nFinal Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt_final))

#Ağacı görselleştirme

plt.figure(figsize=(20, 10))

plot_tree(
    final_dt,
    feature_names=X.columns,
    class_names=["Sağlıklı", "Parkinson"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree - Parkinson Sınıflandırması")
plt.show()

feature_importance = pd.Series(
    final_dt.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

top_features = feature_importance.tail(10)

plt.figure(figsize=(8, 4))

top_features.plot(kind="barh",color="#BA124D")

plt.title("Decision Tree - En Önemli 10 Özellik")
plt.xlabel("Önem Değeri")
plt.ylabel("Özellik")

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------
svm_model = SVC(kernel="rbf", random_state=42)

svm_model.fit(x_train, y_train.values.ravel())

y_pred_svm = svm_model.predict(x_test)

print("SVM Test Tahminleri:")
print(y_pred_svm)

print("\nSVM Accuracy:")
print(accuracy_score(y_test, y_pred_svm))

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))


c_degerleri = [0.01, 0.1, 1, 10, 100]
c_scores = []

for c in c_degerleri:

    svm_model = SVC(
        kernel="rbf",
        C=c
    )

    scores = cross_val_score(
        svm_model,
        x_train,
        y_train.values.ravel(),
        cv=5,
        scoring="accuracy"
    )

    ortalama_accuracy = scores.mean()

    c_scores.append(ortalama_accuracy)

    print(f"C={c}, Ortalama Accuracy={ortalama_accuracy:.3f}")


plt.figure(figsize=(8, 5))

plt.plot(
    c_degerleri,
    c_scores,
    color="#99023E",
    marker="o"
)

plt.xscale("log")

plt.xlabel("C Değeri")
plt.ylabel("Ortalama Accuracy")
plt.title("SVM - C Değeri ve Cross Validation Accuracy")

plt.grid(True)

plt.show()

gamma_degerleri = ["scale", "auto", 0.001, 0.01, 0.1, 1]
gamma_scores = []

for gamma in gamma_degerleri:

    svm_model = SVC(
        kernel="rbf",
        C=100,
        gamma=gamma
    )

    scores = cross_val_score(
        svm_model,
        x_train,
        y_train.values.ravel(),
        cv=5,
        scoring="accuracy"
    )

    ortalama_accuracy = scores.mean()

    gamma_scores.append(ortalama_accuracy)

    print(f"Gamma={gamma}, Ortalama Accuracy={ortalama_accuracy:.3f}")


plt.figure(figsize=(8, 5))

plt.bar(
    [str(g) for g in gamma_degerleri],
      gamma_scores,
    color="#520932"
   
)

plt.xlabel("Gamma Değeri")
plt.ylabel("Ortalama Accuracy")
plt.title("SVM - Gamma Değeri ve Cross Validation Accuracy")

plt.ylim(0.75, 1.00)

plt.grid(axis="y")

plt.show()

final_svm = SVC(
    kernel="rbf",
    C=100,
    gamma=0.1,
    random_state=42
)

final_svm.fit(
    x_train,
    y_train.values.ravel()
)

y_pred_svm_final = final_svm.predict(x_test)

print("Final SVM Test Tahminleri:")
print(y_pred_svm_final)

print("\nFinal SVM Accuracy:")
print(accuracy_score(y_test, y_pred_svm_final))

print("\nFinal SVM Classification Report:")
print(classification_report(y_test, y_pred_svm_final))

print("\nFinal SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm_final))

sonuclar = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "SVM"
    ],
    "Accuracy": [
        0.90,
        0.9744,
        0.9231,
        0.9487
    ],
    "Parkinson Recall": [
        1.00,
        0.97,
        0.97,
        1.00
    ],
    "Parkinson F1": [
        0.94,
        0.98,
        0.95,
        0.97
    ]
})

print("\nModel Karşılaştırması:")
print(sonuclar)

plt.figure(figsize=(10, 6))

plt.bar(
    sonuclar["Model"],
    sonuclar["Accuracy"],
    color=["#741C4A", "#451527", "#782077", "#B310A5"]
)

plt.title("Modellerin Accuracy Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

metrikler = sonuclar.set_index("Model")[
    ["Parkinson Recall", "Parkinson F1"]
]

metrikler.plot(
    kind="bar",
    figsize=(10, 6),
    color=["#B340B3", "#96274F"]
)

plt.title("Parkinson Sınıfı İçin Model Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("Skor")

plt.ylim(0, 1.1)

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------

cm_lr = confusion_matrix(y_test, y_pred)
cm_knn = confusion_matrix(y_test, y_pred_final)
cm_dt = confusion_matrix(y_test, y_pred_dt_final)
cm_svm = confusion_matrix(y_test, y_pred_svm)


plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
sbn.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Sağlıklı", "Parkinson"],
    yticklabels=["Sağlıklı", "Parkinson"]
)
plt.title("Logistic Regression")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")


plt.subplot(1, 4, 2)
sbn.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Sağlıklı", "Parkinson"],
    yticklabels=["Sağlıklı", "Parkinson"]
)
plt.title("KNN")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")


plt.subplot(1, 4, 3)
sbn.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Sağlıklı", "Parkinson"],
    yticklabels=["Sağlıklı", "Parkinson"]
)
plt.title("Decision Tree")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")


plt.subplot(1, 4, 4)
sbn.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Sağlıklı", "Parkinson"],
    yticklabels=["Sağlıklı", "Parkinson"]
)
plt.title("SVM")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")

plt.tight_layout()
plt.show()