import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


INPUT_CSV = "Mall_Customers.csv"

# Si aucune vraie colonne cible n'existe, laisser None
TARGET_COL = None

# Colonnes identifiants a supprimer avant clustering
ID_COLS = ["CustomerID"]

RANDOM_STATE = 42

# K-means classique
K_CLASSIC = 2

# Recherche automatique de K pour la version amelioree
# Si tu veux garder K fixe, mets USE_AUTO_K = False
USE_AUTO_K = True
K_MIN = 2
K_MAX = 10
K_IMPROVED_FIXED = 4

# Parametres K-means classique
CLASSIC_N_INIT = 10
CLASSIC_MAX_ITER = 300

# Parametres version points frontieres
BOUNDARY_MAX_ITER = 30
BOUNDARY_N_RUNS = 5

# Pourcentage dynamique des points les plus ambigus
# Au debut on corrige plus de points, puis moins
BOUNDARY_PERCENTILE_START = 20
BOUNDARY_PERCENTILE_END = 5

# Tolerance numerique
EPS = 1e-9

# Sauvegarde
OUTPUT_CSV = "resultats_kmeans_boundary_dynamic.csv"


# 1) Chargement
def load_dataset(path):
    """Charge un CSV avec separateur virgule ou point-virgule."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path, sep=";")
    return df


# 2) Nettoyage simple
def clean_dataset(df):
    """Nettoyage simple des donnees."""
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            prefix = col + "="
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(rf"^{re.escape(prefix)}", "", regex=True)
                .str.strip()
            )

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

    return df


# 3) Distances et outils
def euclidean_squared_distances(X, centers):
    """Distances euclidiennes au carre entre points et centres."""
    return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def compute_centers_from_labels(X, labels, k, old_centers=None):
    """Recalcule les centres a partir des labels."""
    centers = []
    for cluster_id in range(k):
        points = X[labels == cluster_id]
        if len(points) == 0:
            if old_centers is not None:
                centers.append(old_centers[cluster_id])
            else:
                centers.append(X[np.random.randint(len(X))])
        else:
            centers.append(points.mean(axis=0))
    return np.array(centers)


def compute_inertia(X, labels, centers):
    """Calcule l'inertie."""
    distances = euclidean_squared_distances(X, centers)
    return distances[np.arange(len(X)), labels].sum()


def distance_intra_cluster(X, labels, centers):
    """Distance intra-classe moyenne."""
    distances = np.sqrt(euclidean_squared_distances(X, centers))
    return distances[np.arange(len(X)), labels].mean()


def distance_inter_clusters(centers):
    """Distance inter-classe moyenne entre centres."""
    k = len(centers)
    if k <= 1:
        return 0.0

    dists = []
    for i in range(k):
        for j in range(i + 1, k):
            d = np.sqrt(((centers[i] - centers[j]) ** 2).sum())
            dists.append(d)

    return float(np.mean(dists))


def cluster_balance_score(labels, k):
    """
    Score d'equilibre simple.
    Plus proche de 1 = plus equilibre.
    """
    counts = pd.Series(labels).value_counts()
    expected_size = len(labels) / k
    min_size = counts.min()
    balance = min_size / expected_size
    return min(balance, 1.0)


# 4) Initialisation aleatoire
def random_initial_centers(X, k, rng):
    """Choisit k centres initiaux aleatoires."""
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


# 5) Cout local de deux clusters
def local_sse(points):
    """
    SSE locale d'un groupe de points.
    Si vide, cout = 0.
    """
    if len(points) == 0:
        return 0.0
    center = points.mean(axis=0)
    return ((points - center) ** 2).sum()


def try_boundary_reassignment(X, labels, idx, cluster_a, cluster_b):
    """
    Tente de deplacer un point frontiere du cluster_a vers cluster_b
    si cela reduit le SSE local sur les deux clusters concernes.
    """
    if cluster_a == cluster_b:
        return False

    # Eviter de vider totalement un cluster
    if np.sum(labels == cluster_a) <= 1:
        return False

    mask_a = labels == cluster_a
    mask_b = labels == cluster_b

    points_a = X[mask_a]
    points_b = X[mask_b]
    x = X[idx]

    old_cost = local_sse(points_a) + local_sse(points_b)

    # Nouveau scenario : retirer x de A, ajouter x a B
    new_points_a = X[np.where(mask_a)[0][np.where(np.where(mask_a)[0] != idx)]]
    new_points_b = np.vstack([points_b, x])

    new_cost = local_sse(new_points_a) + local_sse(new_points_b)

    if new_cost + EPS < old_cost:
        labels[idx] = cluster_b
        return True

    return False


# 6) K-means avec traitement dynamique des points frontieres
def boundary_aware_kmeans(
    X,
    k,
    max_iter=30,
    n_runs=5,
    random_state=42,
    percentile_start=20,
    percentile_end=5
):
    """
    K-means ameliore par traitement dynamique des points frontieres.

    Etapes :
    1. Initialisation aleatoire
    2. Affectation classique
    3. Detection des points ambigus :
       A_i = (d2 - d1) / (d1 + eps)
    4. Seuil dynamique par percentile
    5. Test de reassignment local vers le 2e cluster
       seulement si cela diminue le cout local
    6. Recalcul des centres
    """
    best_result = None
    rng_global = np.random.RandomState(random_state)

    for run in range(n_runs):
        rng = np.random.RandomState(rng_global.randint(0, 10**6))
        centers = random_initial_centers(X, k, rng)

        prev_labels = None
        total_boundary_points = 0
        total_reassignments = 0

        for it in range(max_iter):
            # 1) Affectation classique
            distances_sq = euclidean_squared_distances(X, centers)
            nearest_two = np.argsort(distances_sq, axis=1)[:, :2]

            labels = nearest_two[:, 0].copy()
            second_labels = nearest_two[:, 1].copy()

            # 2) Ambiguite
            d1 = np.sqrt(distances_sq[np.arange(len(X)), labels])
            d2 = np.sqrt(distances_sq[np.arange(len(X)), second_labels])

            ambiguity = (d2 - d1) / (d1 + EPS)

            # 3) Seuil dynamique
            current_percentile = (
                percentile_start
                - (percentile_start - percentile_end) * (it / max(1, max_iter - 1))
            )
            threshold = np.percentile(ambiguity, current_percentile)

            boundary_indices = np.where(ambiguity <= threshold)[0]
            boundary_indices = boundary_indices[np.argsort(ambiguity[boundary_indices])]

            total_boundary_points += len(boundary_indices)

            # 4) Correction locale des points frontieres
            labels_corrected = labels.copy()
            moved_this_iter = 0

            for idx in boundary_indices:
                cluster_a = labels_corrected[idx]
                cluster_b = second_labels[idx]

                moved = try_boundary_reassignment(X, labels_corrected, idx, cluster_a, cluster_b)
                if moved:
                    moved_this_iter += 1

            total_reassignments += moved_this_iter

            # 5) Recalcul des centres
            new_centers = compute_centers_from_labels(X, labels_corrected, k, old_centers=centers)

            # 6) Critere d'arret
            if prev_labels is not None and np.array_equal(labels_corrected, prev_labels) and moved_this_iter == 0:
                labels = labels_corrected
                centers = new_centers
                break

            prev_labels = labels_corrected.copy()
            labels = labels_corrected
            centers = new_centers

        inertia = compute_inertia(X, labels, centers)

        result = {
            "labels": labels,
            "centers": centers,
            "inertia": inertia,
            "iterations": it + 1,
            "boundary_points_total": total_boundary_points,
            "reassignments_total": total_reassignments
        }

        # Choix du meilleur run
        nb_clusters = len(np.unique(labels))
        if nb_clusters > 1 and nb_clusters < len(X):
            sil = silhouette_score(X, labels)
        else:
            sil = -1

        result["silhouette"] = sil

        if best_result is None:
            best_result = result
        else:
            # priorite : silhouette, puis inertie
            if result["silhouette"] > best_result["silhouette"] + EPS:
                best_result = result
            elif abs(result["silhouette"] - best_result["silhouette"]) <= EPS and result["inertia"] < best_result["inertia"]:
                best_result = result

    return best_result


# 7) Evaluation
def evaluate_clustering(X, labels, centers, inertia, elapsed_time, method_name):
    """Calcule les metriques principales."""
    nb_clusters = len(np.unique(labels))

    if nb_clusters > 1 and nb_clusters < len(X):
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    else:
        sil = np.nan
        db = np.nan
        ch = np.nan

    intra = distance_intra_cluster(X, labels, centers)
    inter = distance_inter_clusters(centers)
    ratio = inter / (intra + EPS)
    balance = cluster_balance_score(labels, nb_clusters)

    return {
        "Methode": method_name,
        "Clusters": nb_clusters,
        "Inertie": inertia,
        "Silhouette": sil,
        "Davies_Bouldin": db,
        "Calinski_Harabasz": ch,
        "Distance_intra": intra,
        "Distance_inter": inter,
        "Ratio_inter_intra": ratio,
        "Balance": balance,
        "Temps": elapsed_time
    }


def improvement_lower_is_better(classic, improved):
    if classic == 0:
        return np.nan
    return ((classic - improved) / classic) * 100


def improvement_higher_is_better(classic, improved):
    if classic == 0:
        return np.nan
    return ((improved - classic) / classic) * 100


# 8) Programme principal
df = load_dataset(INPUT_CSV)
df = clean_dataset(df)

print("=" * 78)
print("K-MEANS CLASSIQUE VS K-MEANS AVEC TRAITEMENT DYNAMIQUE DES POINTS FRONTIERES")
print("=" * 78)
print(f"Dataset : {INPUT_CSV}")
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# Suppression de la colonne cible si elle existe
if TARGET_COL is not None and TARGET_COL in df.columns:
    y = df[TARGET_COL].copy()
    X_raw = df.drop(columns=[TARGET_COL])
else:
    y = None
    X_raw = df.copy()

# Suppression des colonnes identifiants
for col in ID_COLS:
    if col in X_raw.columns:
        X_raw = X_raw.drop(columns=[col])

# Encodage
X_encoded = pd.get_dummies(X_raw, drop_first=False)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

print(f"Colonnes apres encodage : {X_encoded.shape[1]}")

print("\nPARAMETRES")
print("-" * 78)
print(f"K-means classique : K fixe = {K_CLASSIC}")
if USE_AUTO_K:
    print(f"K-means frontieres : K automatique entre {K_MIN} et {K_MAX}")
else:
    print(f"K-means frontieres : K fixe = {K_IMPROVED_FIXED}")
print(f"Points frontieres  : percentile dynamique {BOUNDARY_PERCENTILE_START}% -> {BOUNDARY_PERCENTILE_END}%")
print(f"Executions         : {BOUNDARY_N_RUNS}")
print("Idee : corriger localement les points ambigus proches de deux centres.\n")


# 9) K-means classique
start = time.time()

classic_model = KMeans(
    n_clusters=K_CLASSIC,
    init="random",
    n_init=CLASSIC_N_INIT,
    max_iter=CLASSIC_MAX_ITER,
    random_state=RANDOM_STATE
)

classic_labels = classic_model.fit_predict(X_scaled)
time_classic = time.time() - start

metrics_classic = evaluate_clustering(
    X_scaled,
    classic_labels,
    classic_model.cluster_centers_,
    classic_model.inertia_,
    time_classic,
    "K-means classique"
)


# 10) K-means ameliore : points frontieres
scores_by_k = []
best_k = None
best_result = None
best_metrics = None

if USE_AUTO_K:
    k_values = range(K_MIN, min(K_MAX, len(X_scaled) - 1) + 1)
else:
    k_values = [K_IMPROVED_FIXED]

search_start = time.time()

for k in k_values:
    result = boundary_aware_kmeans(
        X_scaled,
        k=k,
        max_iter=BOUNDARY_MAX_ITER,
        n_runs=BOUNDARY_N_RUNS,
        random_state=RANDOM_STATE + k,
        percentile_start=BOUNDARY_PERCENTILE_START,
        percentile_end=BOUNDARY_PERCENTILE_END
    )

    labels = result["labels"]
    centers = result["centers"]
    inertia = result["inertia"]

    nb_clusters = len(np.unique(labels))
    if nb_clusters > 1 and nb_clusters < len(X_scaled):
        sil = silhouette_score(X_scaled, labels)
    else:
        sil = -1

    scores_by_k.append({
        "K": k,
        "Silhouette": sil,
        "Inertie": inertia,
        "BoundaryPoints": result["boundary_points_total"],
        "Reassignments": result["reassignments_total"]
    })

    if best_result is None:
        best_result = result
        best_k = k
    else:
        if sil > best_result["silhouette"] + EPS:
            best_result = result
            best_k = k
        elif abs(sil - best_result["silhouette"]) <= EPS and inertia < best_result["inertia"]:
            best_result = result
            best_k = k

time_improved = time.time() - search_start

best_metrics = evaluate_clustering(
    X_scaled,
    best_result["labels"],
    best_result["centers"],
    best_result["inertia"],
    time_improved,
    "K-means frontieres"
)


# 11) Affichage
print("RESULTATS")
print("-" * 78)
print(f"{'Metrique':<30} {'Classique':<20} {'Frontieres':<20}")
print("-" * 78)
print(f"{'K utilise':<30} {K_CLASSIC:<20} {best_k:<20}")
print(f"{'Inertie':<30} {metrics_classic['Inertie']:<20.4f} {best_metrics['Inertie']:<20.4f}")
print(f"{'Silhouette':<30} {metrics_classic['Silhouette']:<20.4f} {best_metrics['Silhouette']:<20.4f}")
print(f"{'Davies-Bouldin':<30} {metrics_classic['Davies_Bouldin']:<20.4f} {best_metrics['Davies_Bouldin']:<20.4f}")
print(f"{'Calinski-Harabasz':<30} {metrics_classic['Calinski_Harabasz']:<20.4f} {best_metrics['Calinski_Harabasz']:<20.4f}")
print(f"{'Distance intra':<30} {metrics_classic['Distance_intra']:<20.4f} {best_metrics['Distance_intra']:<20.4f}")
print(f"{'Distance inter':<30} {metrics_classic['Distance_inter']:<20.4f} {best_metrics['Distance_inter']:<20.4f}")
print(f"{'Ratio inter/intra':<30} {metrics_classic['Ratio_inter_intra']:<20.4f} {best_metrics['Ratio_inter_intra']:<20.4f}")
print(f"{'Balance':<30} {metrics_classic['Balance']:<20.4f} {best_metrics['Balance']:<20.4f}")
print(f"{'Temps total (s)':<30} {metrics_classic['Temps']:<20.4f} {best_metrics['Temps']:<20.4f}")
print(f"{'Points frontieres traites':<30} {'-':<20} {best_result['boundary_points_total']:<20}")
print(f"{'Reaffectations acceptees':<30} {'-':<20} {best_result['reassignments_total']:<20}")

print("\nDETAIL DU CHOIX DE K")
print("-" * 78)
print(f"{'K':<10} {'Silhouette':<15} {'Inertie':<15} {'Frontieres':<15} {'Reaffectations':<15}")
print("-" * 78)
for row in scores_by_k:
    print(
        f"{row['K']:<10} "
        f"{row['Silhouette']:<15.4f} "
        f"{row['Inertie']:<15.4f} "
        f"{row['BoundaryPoints']:<15} "
        f"{row['Reassignments']:<15}"
    )

print("\nDISTRIBUTION DES CLUSTERS")
print("-" * 78)
print("K-means classique :")
print(pd.Series(classic_labels).value_counts().sort_index())

print("\nK-means frontieres :")
print(pd.Series(best_result["labels"]).value_counts().sort_index())


# 12) Pourcentage d'amelioration
print("\nPOURCENTAGE D'AMELIORATION")
print("-" * 78)

inertia_impr = improvement_lower_is_better(metrics_classic["Inertie"], best_metrics["Inertie"])
sil_impr = improvement_higher_is_better(metrics_classic["Silhouette"], best_metrics["Silhouette"])
db_impr = improvement_lower_is_better(metrics_classic["Davies_Bouldin"], best_metrics["Davies_Bouldin"])
ch_impr = improvement_higher_is_better(metrics_classic["Calinski_Harabasz"], best_metrics["Calinski_Harabasz"])
intra_impr = improvement_lower_is_better(metrics_classic["Distance_intra"], best_metrics["Distance_intra"])
inter_impr = improvement_higher_is_better(metrics_classic["Distance_inter"], best_metrics["Distance_inter"])
ratio_impr = improvement_higher_is_better(metrics_classic["Ratio_inter_intra"], best_metrics["Ratio_inter_intra"])
balance_impr = improvement_higher_is_better(metrics_classic["Balance"], best_metrics["Balance"])
time_impr = improvement_lower_is_better(metrics_classic["Temps"], best_metrics["Temps"])

print(f"Reduction inertie              : {inertia_impr:.2f} %")
print(f"Amelioration silhouette        : {sil_impr:.2f} %")
print(f"Reduction Davies-Bouldin       : {db_impr:.2f} %")
print(f"Amelioration Calinski-Harabasz : {ch_impr:.2f} %")
print(f"Reduction distance intra       : {intra_impr:.2f} %")
print(f"Augmentation distance inter    : {inter_impr:.2f} %")
print(f"Amelioration ratio inter/intra : {ratio_impr:.2f} %")
print(f"Amelioration balance           : {balance_impr:.2f} %")
print(f"Reduction temps                : {time_impr:.2f} %")


# 13) Sauvegarde des resultats
df_results = df.copy()
df_results["Cluster_KMeans_Classique"] = classic_labels
df_results["Cluster_KMeans_Frontieres"] = best_result["labels"]
df_results.to_csv(OUTPUT_CSV, index=False)

print(f"\nFichier sauvegarde : {OUTPUT_CSV}")


# 14) Visualisations
scores_df = pd.DataFrame(scores_by_k)

plt.figure(figsize=(8, 5))
plt.plot(scores_df["K"], scores_df["Silhouette"], marker="o")
plt.xlabel("Nombre de clusters K")
plt.ylabel("Score silhouette")
plt.title("Choix de K - K-means avec points frontieres")
plt.xticks(scores_df["K"])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(scores_df["K"], scores_df["BoundaryPoints"], marker="o", label="Points frontieres")
plt.plot(scores_df["K"], scores_df["Reassignments"], marker="o", label="Reaffectations")
plt.xlabel("Nombre de clusters K")
plt.ylabel("Nombre")
plt.title("Activite de correction des points frontieres")
plt.xticks(scores_df["K"])
plt.legend()
plt.tight_layout()
plt.show()

labels_bar = ["Silhouette", "Intra", "Inter", "Ratio", "Balance"]
classic_values = [
    metrics_classic["Silhouette"],
    metrics_classic["Distance_intra"],
    metrics_classic["Distance_inter"],
    metrics_classic["Ratio_inter_intra"],
    metrics_classic["Balance"]
]
improved_values = [
    best_metrics["Silhouette"],
    best_metrics["Distance_intra"],
    best_metrics["Distance_inter"],
    best_metrics["Ratio_inter_intra"],
    best_metrics["Balance"]
]

x = np.arange(len(labels_bar))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, classic_values, width, label="Classique")
plt.bar(x + width / 2, improved_values, width, label="Frontieres")
plt.xticks(x, labels_bar)
plt.title("Comparaison K-means classique vs K-means points frontieres")
plt.legend()
plt.tight_layout()
plt.show()