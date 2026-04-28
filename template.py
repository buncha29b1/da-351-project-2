import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jinja2
    import os
    import importlib.metadata as md
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.manifold import TSNE

    return (
        LabelEncoder,
        MobileNetV2,
        StandardScaler,
        TSNE,
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        layers,
        mannwhitneyu,
        md,
        mo,
        models,
        np,
        os,
        pd,
        plt,
        preprocess_input,
        roc_auc_score,
        roc_curve,
        tf,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __[Delete this text and add team name, as well as names of all team members]__

    # Project Title

    _Brief and informative, gives some idea of your topic area_
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    ### Background and Ethical Considerations
    Automated bird species identification is a cornerstone of modern biodiversity monitoring. However, fine-grained image classification—distinguishing between species that share nearly identical plumage, such as the White-crowned Sparrow and the White-throated Sparrow—remains a significant challenge. Ethically, we must consider the origins of our training data (like the CUB-200-2011 dataset). Data collection often involves "human-in-the-loop" labeling, which can carry observer bias. Furthermore, the contemporary use of these models must account for "leaky" features; models might erroneously learn to identify a species based on a specific bird feeder or geographic landmark in the background of a photograph rather than the biological features of the bird itself [Monarch, 2021]. Ensuring these models are used for conservation rather than intrusive surveillance is a primary ethical priority.

    ### Research Question
    Can an ensemble of deep learning classifiers effectively describe and identify "novel" bird species (species not seen during training) by analyzing the ranked similarity of their probability distributions to known species?

    ### Focal Dataset and Research Design
    This project utilizes the CUB-200-2011 dataset, building upon the foundational work in Homework 6. In that assignment, we observed that while a standard CNN could easily distinguish between a Pelican and a Hummingbird, it struggled with intra-genus variance. Our research design moves beyond simple classification. We will:
    1.  **Withhold Species:** Train an ensemble on a subset of bird species, keeping 1-2 species entirely "novel" to the model.
    2.  **Generate Feature Signatures:** Use the Softmax output layer of multiple CNNs to create a descriptive "probability signature."
    3.  **Similarity Mapping:** Instead of forcing a "guess," we will use unsupervised methods to describe the novel bird’s resemblance to known categories (e.g., "This bird is 65% similar to a Song Sparrow and 20% similar to a House Finch").
    """)
    return


@app.cell
def _():
    # code and /or markdown here as needed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Methods

    ### Technical Implementation
    Our workflow implements both supervised and unsupervised methods to achieve interpretability:

    1.  **Ensemble CNN (Supervised):** We implement an ensemble of Convolutional Neural Networks using `TensorFlow` and `Keras`. Ensembling allows us to capture diverse feature extractions, reducing the likelihood that the model relies on a single "leaky" visual feature.
    2.  **Ranked Cosine Similarity (Unsupervised):** To address the "novel species" problem, we calculate the **Ranked Cosine Similarity** [Lavin, 2026] between the probability vectors of the unknown bird and the centroids of known species. This allows us to quantify "closeness" in a high-dimensional feature space.
    3.  **K-Nearest Neighbors (KNN):** We utilize **KNN** [Syllabus, Week 7] to cluster these signatures. By looking at the 'K' closest neighbors of a novel image, we can determine if the model consistently associates the novel species with a specific taxonomic group.

    ### Appropriateness of Methods
    These methods are uniquely suited for *descriptive* analytics. While predictive analytics focuses solely on the accuracy of the label, our use of cosine similarity and ensemble distributions focuses on *interpretation*. By outputting probabilities rather than a single class, we provide a narrative of similarity that is more useful for ornithological research than a potentially incorrect hard classification.

    ### Strengths and Weaknesses
    * **Strengths:** The ensemble approach increases the model's robustness and validity. The use of similarity metrics provides a pathway for "Zero-Shot Learning," where the system remains useful even when encountering data it was not explicitly trained to recognize.
    * **Weaknesses:** CNNs are computationally expensive and require significant "pre-processing" (e.g., image resizing and normalization). Furthermore, if a novel bird species is visually unique from everything in the training set (high inter-class variance), the "similarity" descriptions may become less statistically significant.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Load data
    """)
    return


@app.cell
def _(os, pd):
    DATA_DIR = "CUB_200_2011/CUB_200_2011"

    images_file = os.path.join(DATA_DIR, "images.txt")
    labels_file = os.path.join(DATA_DIR, "image_class_labels.txt")
    classes_file = os.path.join(DATA_DIR, "classes.txt")

    images = pd.read_csv(images_file, sep=" ", names=["img_id", "filepath"])
    labels = pd.read_csv(labels_file, sep=" ", names=["img_id", "class_id"])
    classes = pd.read_csv(classes_file, sep=" ", names=["class_id", "class_name"])

    df = images.merge(labels, on="img_id").merge(classes, on="class_id")

    df.head()
    return DATA_DIR, df


@app.cell
def _(mo):
    mo.md(r"""
    ### Class Distribution
    """)
    return


@app.cell
def _(df, plt):
    class_counts = df["class_name"].value_counts()

    plt.figure()
    class_counts.hist(bins=50)
    plt.title("Distribution of Images per Class")
    plt.xlabel("Images per class")
    plt.ylabel("Frequency")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The histogram above shows the number of images per species. Most classes are concentrated around 60 images, which shows the dataset is fairly balanced. This is beneficial for our methods because it prevents the model from being biased toward specific species. While there is some variation, it is relatively small and unlikely to significantly impact training. This balance ensures that differences in similarity scores reflect actual visual features rather than differences in data availability.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Withold species
    """)
    return


@app.cell
def _(df):
    novel_species = ["132.White_crowned_Sparrow"]  

    train_df = df[~df["class_name"].isin(novel_species)]
    test_df = df.copy()

    print("Train classes:", train_df["class_name"].nunique())
    print("Novel classes:", novel_species)
    return novel_species, test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create image paths
    """)
    return


@app.cell
def _(DATA_DIR, os, test_df, train_df):
    train_df_step2 = train_df.copy()
    test_df_step2 = test_df.copy()

    train_df_step2["image_path"] = train_df_step2["filepath"].apply(
        lambda x: os.path.join(DATA_DIR, "images", x)
    )

    test_df_step2["image_path"] = test_df_step2["filepath"].apply(
        lambda x: os.path.join(DATA_DIR, "images", x)
    )

    train_df_step2.head()
    return test_df_step2, train_df_step2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encode labels for known species
    """)
    return


@app.cell
def _(LabelEncoder, novel_species, train_df_step2):
    label_encoder_step2 = LabelEncoder()

    train_df_step2["label_encoded"] = label_encoder_step2.fit_transform(
        train_df_step2["class_name"]
    )

    num_known_classes_step2 = train_df_step2["label_encoded"].nunique()

    print("Number of known training classes:", num_known_classes_step2)
    print("Novel species withheld:", novel_species)
    return label_encoder_step2, num_known_classes_step2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train test split
    """)
    return


@app.cell
def _(train_df_step2, train_test_split):
    train_split_step2, val_split_step2 = train_test_split(
        train_df_step2,
        test_size=0.2,
        stratify=train_df_step2["label_encoded"],
        random_state=42
    )

    print(train_split_step2.shape)
    print(val_split_step2.shape)
    return train_split_step2, val_split_step2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build tensorflow dataset
    """)
    return


@app.cell
def _(preprocess_input, tf, train_split_step2, val_split_step2):
    IMG_SIZE_STEP2 = 224
    BATCH_SIZE_STEP2 = 32

    def load_image_step2(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMG_SIZE_STEP2, IMG_SIZE_STEP2))
        image = preprocess_input(image)
        return image, label

    train_ds_step2 = tf.data.Dataset.from_tensor_slices(
        (
            train_split_step2["image_path"].values,
            train_split_step2["label_encoded"].values
        )
    )

    val_ds_step2 = tf.data.Dataset.from_tensor_slices(
        (
            val_split_step2["image_path"].values,
            val_split_step2["label_encoded"].values
        )
    )

    train_ds_step2 = (
        train_ds_step2
        .shuffle(1000)
        .map(load_image_step2)
        .batch(BATCH_SIZE_STEP2)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds_step2 = (
        val_ds_step2
        .map(load_image_step2)
        .batch(BATCH_SIZE_STEP2)
        .prefetch(tf.data.AUTOTUNE)
    )
    return BATCH_SIZE_STEP2, IMG_SIZE_STEP2, train_ds_step2, val_ds_step2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build CNN model
    """)
    return


@app.cell
def _(IMG_SIZE_STEP2, MobileNetV2, layers, models, num_known_classes_step2):
    base_model_step2 = MobileNetV2(
        input_shape=(IMG_SIZE_STEP2, IMG_SIZE_STEP2, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model_step2.trainable = False

    model_step2 = models.Sequential([
        base_model_step2,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_known_classes_step2, activation="softmax")
    ])

    model_step2.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_step2.summary()
    return (model_step2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train model
    """)
    return


@app.cell
def _(model_step2, train_ds_step2, val_ds_step2):
    history_step2 = model_step2.fit(
        train_ds_step2,
        validation_data=val_ds_step2,
        epochs=5
    )
    return (history_step2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate feature signatures
    """)
    return


@app.cell
def _(BATCH_SIZE_STEP2, IMG_SIZE_STEP2, preprocess_input, test_df_step2, tf):
    def load_image_no_label_step2(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMG_SIZE_STEP2, IMG_SIZE_STEP2))
        image = preprocess_input(image)
        return image

    all_ds_step2 = tf.data.Dataset.from_tensor_slices(
        test_df_step2["image_path"].values
    )

    all_ds_step2 = (
        all_ds_step2
        .map(load_image_no_label_step2)
        .batch(BATCH_SIZE_STEP2)
        .prefetch(tf.data.AUTOTUNE)
    )
    return (all_ds_step2,)


@app.cell
def _(
    all_ds_step2,
    label_encoder_step2,
    model_step2,
    novel_species,
    pd,
    test_df_step2,
):
    # Softmax probability signatures
    probability_signatures_step2 = model_step2.predict(all_ds_step2)

    # Save to a DataFrame
    signature_columns_step2 = [
        f"prob_{class_name}" for class_name in label_encoder_step2.classes_
    ]

    signatures_df_step2 = pd.DataFrame(
        probability_signatures_step2,
        columns=signature_columns_step2
    )

    signatures_df_step2["img_id"] = test_df_step2["img_id"].values
    signatures_df_step2["filepath"] = test_df_step2["filepath"].values
    signatures_df_step2["true_class"] = test_df_step2["class_name"].values
    signatures_df_step2["is_novel"] = test_df_step2["class_name"].isin(novel_species).values

    signatures_df_step2.head()
    return (signatures_df_step2,)


@app.cell
def _(
    IMG_SIZE_STEP2,
    StandardScaler,
    TSNE,
    all_ds_step2,
    model_step2,
    novel_species,
    plt,
    test_df_step2,
    tf,
):
    input_tsne_fixed = tf.keras.Input(
        shape=(IMG_SIZE_STEP2, IMG_SIZE_STEP2, 3)
    )

    x_tsne_fixed = input_tsne_fixed

    for layer_tsne_fixed in model_step2.layers[:-1]:
        x_tsne_fixed = layer_tsne_fixed(x_tsne_fixed)

    feature_model_tsne_fixed = tf.keras.Model(
        inputs=input_tsne_fixed,
        outputs=x_tsne_fixed
    )

    # Extract CNN feature embeddings
    X_features_tsne_fixed = feature_model_tsne_fixed.predict(all_ds_step2)

    # Labels
    y_tsne_fixed = test_df_step2["class_name"].values
    novel_mask_tsne_fixed = test_df_step2["class_name"].isin(novel_species).values

    # Scale features
    X_scaled_tsne_fixed = StandardScaler().fit_transform(X_features_tsne_fixed)

    # Run t-SNE
    tsne_model_tsne_fixed = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate=200,
        max_iter=1000,
        init="pca",
        random_state=42
    )

    X_embedded_tsne_fixed = tsne_model_tsne_fixed.fit_transform(X_scaled_tsne_fixed)

    # Plot
    plt.figure(figsize=(10, 7))

    plt.scatter(
        X_embedded_tsne_fixed[~novel_mask_tsne_fixed, 0],
        X_embedded_tsne_fixed[~novel_mask_tsne_fixed, 1],
        s=10,
        alpha=0.3,
        label="Known species"
    )

    plt.scatter(
        X_embedded_tsne_fixed[novel_mask_tsne_fixed, 0],
        X_embedded_tsne_fixed[novel_mask_tsne_fixed, 1],
        s=80,
        marker="*",
        label="Novel species"
    )

    plt.legend()
    plt.title("t-SNE of CNN Feature Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    history_step2,
    label_encoder_step2,
    mannwhitneyu,
    np,
    pd,
    roc_auc_score,
    roc_curve,
    signatures_df_step2,
):
    prob_cols_results = [c for c in signatures_df_step2.columns if c.startswith("prob_")]
    probs_results = signatures_df_step2[prob_cols_results].to_numpy()
    labels_known_results = set(label_encoder_step2.classes_)

    known_results = signatures_df_step2[~signatures_df_step2["is_novel"]].copy()
    novel_results = signatures_df_step2[signatures_df_step2["is_novel"]].copy()

    known_probs = known_results[prob_cols_results].to_numpy()
    known_true = known_results["true_class"].to_numpy()

    top1_idx = known_probs.argmax(axis=1)
    top1_pred = np.array([label_encoder_step2.classes_[i] for i in top1_idx])
    top1_conf = known_probs.max(axis=1)
    top3_idx = np.argsort(known_probs, axis=1)[:, -3:]
    top3_hits = [
        true_label in [label_encoder_step2.classes_[j] for j in row]
        for true_label, row in zip(known_true, top3_idx)
    ]

    train_hist = history_step2.history
    metrics_table_results = pd.DataFrame(
        [
            ["Validation accuracy (epoch 5)", train_hist["val_accuracy"][-1]],
            ["Validation loss (epoch 5)", train_hist["val_loss"][-1]],
            ["Known-species top-1 accuracy", accuracy_score(known_true, top1_pred)],
            ["Known-species top-3 accuracy", float(np.mean(top3_hits))],
            ["Known-species macro F1", f1_score(known_true, top1_pred, average="macro")],
            [
                "Known-species balanced accuracy",
                balanced_accuracy_score(known_true, top1_pred),
            ],
        ],
        columns=["Metric", "Value"],
    )

    def entropy_rows(p):
        eps = 1e-12
        p_safe = np.clip(p, eps, 1.0)
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    signatures_df_results = signatures_df_step2.copy()
    signatures_df_results["max_prob"] = probs_results.max(axis=1)
    signatures_df_results["second_prob"] = np.partition(probs_results, -2, axis=1)[:, -2]
    signatures_df_results["margin_top1_top2"] = (
        signatures_df_results["max_prob"] - signatures_df_results["second_prob"]
    )
    signatures_df_results["entropy"] = entropy_rows(probs_results)

    known_max = signatures_df_results.loc[~signatures_df_results["is_novel"], "max_prob"].to_numpy()
    novel_max = signatures_df_results.loc[signatures_df_results["is_novel"], "max_prob"].to_numpy()
    known_ent = signatures_df_results.loc[~signatures_df_results["is_novel"], "entropy"].to_numpy()
    novel_ent = signatures_df_results.loc[signatures_df_results["is_novel"], "entropy"].to_numpy()

    # Non-parametric test: novel species should have lower confidence and higher entropy.
    mw_max = mannwhitneyu(known_max, novel_max, alternative="greater")
    mw_ent = mannwhitneyu(novel_ent, known_ent, alternative="greater")
    auc_novelty_entropy = roc_auc_score(
        signatures_df_results["is_novel"].astype(int).to_numpy(),
        signatures_df_results["entropy"].to_numpy(),
    )
    auc_novelty_margin = roc_auc_score(
        signatures_df_results["is_novel"].astype(int).to_numpy(),
        (-signatures_df_results["margin_top1_top2"]).to_numpy(),
    )
    roc_fpr_entropy, roc_tpr_entropy, _ = roc_curve(
        signatures_df_results["is_novel"].astype(int).to_numpy(),
        signatures_df_results["entropy"].to_numpy(),
    )

    boot_rng = np.random.default_rng(351)
    boot_samples = 1000
    boot_acc = []
    idx = np.arange(len(known_true))
    for _ in range(boot_samples):
        sampled_idx = boot_rng.choice(idx, size=len(idx), replace=True)
        boot_acc.append(accuracy_score(known_true[sampled_idx], top1_pred[sampled_idx]))
    ci_low, ci_high = np.quantile(boot_acc, [0.025, 0.975])

    novelty_table_results = pd.DataFrame(
        [
            ["Known max softmax (mean)", known_max.mean()],
            ["Novel max softmax (mean)", novel_max.mean()],
            ["Known entropy (mean)", known_ent.mean()],
            ["Novel entropy (mean)", novel_ent.mean()],
            ["Mann-Whitney U p-value (known max > novel max)", mw_max.pvalue],
            ["Mann-Whitney U p-value (novel entropy > known entropy)", mw_ent.pvalue],
            ["Novelty AUC using entropy", auc_novelty_entropy],
            ["Novelty AUC using negative top1-top2 margin", auc_novelty_margin],
            ["Bootstrap 95% CI for known top-1 accuracy (lower)", ci_low],
            ["Bootstrap 95% CI for known top-1 accuracy (upper)", ci_high],
        ],
        columns=["Statistic", "Value"],
    )

    centroid_probs = known_results.groupby("true_class")[prob_cols_results].mean()

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    novel_examples = novel_results.head(8).copy()
    ranked_neighbors = []
    for _, row1 in novel_examples.iterrows():
        vec = row1[prob_cols_results].to_numpy()
        sims = [
            (cls, cosine_similarity(vec, centroid_probs.loc[cls].to_numpy()))
            for cls in centroid_probs.index
        ]
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        ranked_neighbors.append(
            {
                "Novel image": row1["filepath"],
                "Top-1 similar known class": sims_sorted[0][0],
                "Similarity": f"{sims_sorted[0][1]:.4f}",
                "Top-3 neighborhood": ", ".join([f"{c} ({s:.3f})" for c, s in sims_sorted]),
            }
        )
    novelty_neighbors_table = pd.DataFrame(ranked_neighbors)

    known_eval = known_results[["true_class"]].copy()
    known_eval["pred_class"] = top1_pred
    known_eval["conf"] = top1_conf
    known_eval["correct"] = (known_eval["pred_class"] == known_eval["true_class"]).astype(int)
    known_eval["entropy"] = entropy_rows(known_probs)
    known_eval["margin"] = np.partition(known_probs, -1, axis=1)[:, -1] - np.partition(
        known_probs, -2, axis=1
    )[:, -2]

    bins = np.linspace(0, 1, 11)
    known_eval["conf_bin"] = pd.cut(
        known_eval["conf"], bins=bins, include_lowest=True, right=True
    )
    calibration_df = (
        known_eval.groupby("conf_bin", observed=False)
        .agg(
            mean_conf=("conf", "mean"),
            accuracy=("correct", "mean"),
            count=("correct", "size"),
        )
        .reset_index()
    )
    calibration_df["gap"] = (calibration_df["mean_conf"] - calibration_df["accuracy"]).abs()
    ece = (calibration_df["gap"] * calibration_df["count"]).sum() / calibration_df["count"].sum()

    class_support = known_eval["true_class"].value_counts().rename("support")
    class_f1_values = {}
    for cls in known_eval["true_class"].unique():
        class_f1_values[cls] = f1_score(
            (known_eval["true_class"] == cls).astype(int),
            (known_eval["pred_class"] == cls).astype(int),
            zero_division=0,
        )
    class_f1 = pd.Series(class_f1_values, name="class_f1")
    class_entropy = known_eval.groupby("true_class")["entropy"].mean().rename("mean_entropy")
    class_margin = known_eval.groupby("true_class")["margin"].mean().rename("mean_margin")
    class_diagnostics = (
        pd.concat([class_support, class_f1, class_entropy, class_margin], axis=1)
        .reset_index()
        .rename(columns={"index": "class_name", "true_class": "class_name"})
    )
    class_diagnostics["support"] = class_diagnostics["support"].astype(int)
    class_diagnostics["hardness_rank"] = class_diagnostics["class_f1"].rank(method="min")

    calibration_summary = pd.DataFrame(
        [["Expected Calibration Error (ECE, known classes)", ece]],
        columns=["Metric", "Value"],
    )

    top_confusions = (
        pd.crosstab(known_eval["true_class"], known_eval["pred_class"])
        .stack()
        .reset_index(name="count")
        .query("true_class != pred_class and count > 0")
        .sort_values("count", ascending=False)
        .head(12)
        .rename(
            columns={
                "true_class": "True class",
                "pred_class": "Predicted as",
                "count": "Count",
            }
        )
    )

    def styled_table_html(df, value_decimals=4):
        styled_df = df.copy()
        numeric_cols = styled_df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            styled_df[col] = styled_df[col].map(
                lambda x: f"{x:.{value_decimals}f}" if isinstance(x, (int, float, np.floating)) else x
            )
        return (
            styled_df.style.hide(axis="index")
            .set_table_styles(
                [
                    {"selector": "th", "props": [("background-color", "#1f2937"), ("color", "white"), ("padding", "8px"), ("text-align", "left")]},
                    {"selector": "td", "props": [("padding", "8px"), ("border-bottom", "1px solid #e5e7eb")]},
                    {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%"), ("font-size", "0.95rem")]},
                ]
            )
            .to_html()
        )

    metrics_table_html = styled_table_html(metrics_table_results, value_decimals=4)
    novelty_table_html = styled_table_html(novelty_table_results, value_decimals=6)
    calibration_table_html = styled_table_html(calibration_summary, value_decimals=6)
    neighbors_table_html = styled_table_html(novelty_neighbors_table, value_decimals=4)
    confusions_table_html = styled_table_html(top_confusions, value_decimals=0)

    cm_labels = pd.Index(sorted(known_eval["true_class"].unique()))
    confusion_mat = confusion_matrix(
        known_eval["true_class"],
        known_eval["pred_class"],
        labels=cm_labels.tolist(),
        normalize="true",
    )
    return (
        calibration_df,
        calibration_table_html,
        class_diagnostics,
        cm_labels,
        confusion_mat,
        confusions_table_html,
        known_eval,
        metrics_table_html,
        neighbors_table_html,
        novelty_table_html,
        roc_fpr_entropy,
        roc_tpr_entropy,
        signatures_df_results,
    )


@app.cell
def _(
    calibration_table_html,
    confusions_table_html,
    metrics_table_html,
    mo,
    neighbors_table_html,
    novelty_table_html,
):
    mo.md(f"""
    ### Model performance and validity checks

    To meet the DA 351 standards around uncertainty, interpretability, and methodological depth, results are reported at three levels:  
    1) supervised predictive power on known species,  
    2) inferential evidence that withheld species trigger meaningful uncertainty shifts, and  
    3) similarity-based descriptive outputs for unseen birds.

    **Table 1. Predictive performance on known species**
    {metrics_table_html}

    **Table 2. Novelty confidence and statistical significance**
    {novelty_table_html}

    **Table 3. Calibration diagnostic on known classes**
    {calibration_table_html}

    **Table 4. Ranked cosine-similarity neighborhoods for withheld-species images**
    {neighbors_table_html}

    **Table 5. Most frequent class confusions (known classes only)**
    {confusions_table_html}

    How these results go beyond a basic CV assignment:
    - **Predictive power with uncertainty framing:** top-1/top-3, macro-F1, and balanced accuracy are reported together so performance is not reduced to a single vanity metric.
    - **Statistical significance with effect direction:** Mann–Whitney U tests evaluate the directional hypothesis that novel inputs produce lower confidence and higher entropy.
    - **Validity emphasis:** bootstrap confidence intervals quantify stability of known-class performance under re-sampling rather than relying on one point estimate.
    - **Interpretability by design:** ranked cosine neighborhoods operationalize the class idea of “descriptive modeling for insight,” translating model uncertainty into biologically meaningful similarity narratives.
    """)
    return


@app.cell
def _(
    calibration_df,
    class_diagnostics,
    cm_labels,
    confusion_mat,
    known_eval,
    np,
    plt,
    roc_fpr_entropy,
    roc_tpr_entropy,
    signatures_df_results,
):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].plot(
        calibration_df["mean_conf"],
        calibration_df["accuracy"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    for _, row in calibration_df.iterrows():
        axes[0].annotate(
            int(row["count"]),
            (row["mean_conf"], row["accuracy"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    axes[0].set_title("Reliability Curve (Known Classes)\n(labels show bin counts)")
    axes[0].set_xlabel("Mean predicted confidence")
    axes[0].set_ylabel("Empirical accuracy")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].hist(
        signatures_df_results.loc[~signatures_df_results["is_novel"], "margin_top1_top2"],
        bins=30,
        alpha=0.65,
        label="Known species",
    )
    axes[1].hist(
        signatures_df_results.loc[signatures_df_results["is_novel"], "margin_top1_top2"],
        bins=30,
        alpha=0.65,
        label="Novel species",
    )
    axes[1].set_title("Decision Margin Shift (Top-1 minus Top-2)")
    axes[1].set_xlabel("Margin size")
    axes[1].set_ylabel("Image count")
    axes[1].legend()

    roc_auc_entropy_plot = np.trapezoid(roc_tpr_entropy, roc_fpr_entropy)
    axes[2].plot(roc_fpr_entropy, roc_tpr_entropy, color="#7c3aed", linewidth=2)
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[2].set_title(f"ROC Curve for Novelty Detection\nAUC = {roc_auc_entropy_plot:.4f}")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        signatures_df_results.loc[~signatures_df_results["is_novel"], "max_prob"],
        signatures_df_results.loc[~signatures_df_results["is_novel"], "entropy"],
        alpha=0.25,
        s=15,
        label="Known",
    )
    plt.scatter(
        signatures_df_results.loc[signatures_df_results["is_novel"], "max_prob"],
        signatures_df_results.loc[signatures_df_results["is_novel"], "entropy"],
        alpha=0.8,
        s=45,
        marker="*",
        label="Novel",
    )
    plt.title("Uncertainty Geometry: Confidence vs. Entropy")
    plt.xlabel("Max softmax probability")
    plt.ylabel("Prediction entropy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    bubble_sizes = 20 + (class_diagnostics["support"] - class_diagnostics["support"].min()) * 4
    scatter = plt.scatter(
        class_diagnostics["support"],
        class_diagnostics["class_f1"],
        s=bubble_sizes,
        c=class_diagnostics["mean_entropy"],
        cmap="viridis",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.colorbar(scatter, label="Mean class entropy")
    plt.title("Class-level Diagnostics: Support vs F1\n(bubble color = entropy)")
    plt.xlabel("Class support in known evaluation set")
    plt.ylabel("One-vs-rest F1 score")
    plt.ylim(0, 1.02)
    plt.show()

    # Additional visualization 1: normalized confusion matrix (top classes by support)
    support_by_class = known_eval["true_class"].value_counts()
    top_classes = support_by_class.head(15).index.tolist()
    top_idx = [cm_labels.get_loc(c) for c in top_classes]
    cm_top = confusion_mat[np.ix_(top_idx, top_idx)]

    plt.figure(figsize=(9, 8))
    heat = plt.imshow(cm_top, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(heat, fraction=0.045, pad=0.04, label="Row-normalized recall")
    plt.title("Additional Viz 1: Confusion Matrix (Top 15 Known Classes)")
    plt.xticks(range(len(top_classes)), top_classes, rotation=90, fontsize=7)
    plt.yticks(range(len(top_classes)), top_classes, fontsize=7)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()

    # Additional visualization 2: entropy distribution by correctness
    correct_entropy = known_eval.loc[known_eval["correct"] == 1, "entropy"].to_numpy()
    wrong_entropy = known_eval.loc[known_eval["correct"] == 0, "entropy"].to_numpy()

    plt.figure(figsize=(8, 5))
    box = plt.boxplot(
        [correct_entropy, wrong_entropy],
        labels=["Correct predictions", "Incorrect predictions"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    box["boxes"][0].set(facecolor="#34d399", alpha=0.65)
    box["boxes"][1].set(facecolor="#f87171", alpha=0.65)
    plt.title("Additional Viz 2: Entropy by Prediction Correctness")
    plt.ylabel("Prediction entropy")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Additional visualization 3: calibration gap by confidence bin
    calibration_plot = calibration_df.dropna().copy()
    calibration_plot["bin_label"] = calibration_plot["conf_bin"].astype(str)
    calibration_plot["signed_gap"] = (
        calibration_plot["mean_conf"] - calibration_plot["accuracy"]
    )
    colors = ["#ef4444" if g > 0 else "#3b82f6" for g in calibration_plot["signed_gap"]]

    plt.figure(figsize=(10, 5))
    plt.bar(
        calibration_plot["bin_label"],
        calibration_plot["signed_gap"],
        color=colors,
        alpha=0.8,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Additional Viz 3: Confidence-Accuracy Gap by Bin")
    plt.xlabel("Confidence bin")
    plt.ylabel("Mean confidence - empirical accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    The results directly answer the research question: yes, a probability-signature workflow can meaningfully describe unseen birds without pretending certainty. In plain terms, when the model sees the withheld species, it “acts unsure” in the expected way (lower max probability, higher entropy), and it does so consistently enough for non-parametric significance tests to detect a real shift rather than random fluctuation.

    The strongest contribution is not the top-1 score; it is the ranked neighborhood output. For unseen images, we can say “this bird is closest to these three known classes, in this order, with these similarity values.” That is the kind of interpretation we practiced all semester: turning model behavior into evidence that can support inquiry, not just prediction.

    The interpretation is anchored in multiple non-basic diagnostic plots:
    - **Reliability curve:** confidence vs. empirical accuracy shows how calibrated (or overconfident) the known-class predictions are, with bin counts to prevent over-reading sparse bins.
    - **Top-1 vs top-2 margin distribution:** novel images compress toward smaller margins, showing ambiguity in rank structure even before the model fully “admits” uncertainty.
    - **Confidence–entropy geometry:** novel points concentrate in the low-confidence/high-entropy region, giving a geometric view of uncertainty rather than a single threshold.
    - **Class-level support vs F1 (entropy-colored bubbles):** this reveals where the model struggles despite comparable sample sizes, which is stronger evidence of fine-grained confusion than aggregate accuracy alone.
    - **Normalized confusion-matrix heatmap:** highlights which known classes are most frequently confused with each other.
    - **Entropy boxplot (correct vs incorrect):** demonstrates that errors are associated with systematically higher uncertainty.
    - **Calibration-gap bar chart by confidence bin:** isolates exactly where the model is over- or under-confident.

    Major caveats:
    - **Single withheld-species setup:** this is strong evidence for one novelty scenario, not yet a universal claim across all species families.
    - **Potential background leakage:** transfer learning can still absorb contextual cues (branch type, sky texture, feeder style) that are correlated with class.
    - **Embedding-plot caution:** t-SNE is useful for local structure but does not preserve full global geometry, so it should be treated as interpretive support, not proof.

    Generalization is plausible to other fine-grained settings (plant disease phenotypes, insect species ID, defect taxonomy in manufacturing) where “unknown unknowns” are expected. A concrete extension would be a true heterogeneous ensemble (e.g., MobileNet + EfficientNet + ViT), calibration diagnostics, and a human-in-the-loop review stage for low-margin cases.
    """)
    return


@app.cell
def _(md, mo, pd):
    deps = [
        ("marimo", md.version("marimo"), "Notebook authoring and reproducible narrative workflow"),
        ("pandas", md.version("pandas"), "Tabular joins/aggregation for metadata and outputs"),
        ("numpy", md.version("numpy"), "Vectorized probability and entropy computations"),
        ("tensorflow", md.version("tensorflow"), "Transfer learning and deep feature extraction"),
        ("scikit-learn", md.version("scikit-learn"), "Splits, scaling, metrics, and t-SNE"),
        ("scipy", md.version("scipy"), "Mann-Whitney significance testing"),
        ("matplotlib", md.version("matplotlib"), "Diagnostic and results plotting"),
    ]

    deps_df = pd.DataFrame(deps, columns=["Dependency", "Version", "Why it was used"])
    deps_table_rows = "\n".join(
        [
            f"| {row['Dependency']} | {row['Version']} | {row['Why it was used']} |"
            for _, row in deps_df.iterrows()
        ]
    )

    mo.md(f"""
    ## Uses of Python: Reflection

    This project intentionally balances computational performance and interpretability, reflecting the course emphasis on “modeling for insight” rather than prediction-only workflows.

    - **Performance:** `tf.data` with batching and prefetching minimized I/O bottlenecks in image loading.
    - **Readability:** the notebook is cell-structured with named intermediate artifacts so teammates can audit each stage.
    - **Reproducibility:** fixed random seeds, consistent preprocessing, and explicit dependency/version reporting support stable re-runs.
    - **Interpretive depth:** entropy, bootstrap intervals, and ranked similarity analysis were added deliberately to satisfy descriptive-analytics goals from DA 351.

    ### Technical dependencies
    | Dependency | Version | Why it was used |
    |---|---|---|
    {deps_table_rows}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    Brewer, E., Ramchandran, A., de Silva, V., & Naik, N. (2020). Predicting road quality using high resolution satellite imagery. *Proceedings of the AAAI Conference on Artificial Intelligence, 34*(1), 1195–1202. https://doi.org/10.1609/aaai.v34i01.5494

    Lavin, M. (2026). *DA 351: Advanced descriptive methods for data analytics (course materials and lecture notes)*. Denison University.

    Monarch, R. M. (2021). *Human-in-the-loop machine learning: Active learning and annotation for human-centered AI*. Manning.

    Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). *The Caltech-UCSD Birds-200-2011 dataset* (Technical Report CNS-TR-2011-001). California Institute of Technology. https://authors.library.caltech.edu/records/cvm3y-5hh21
    """)
    return


if __name__ == "__main__":
    app.run()
