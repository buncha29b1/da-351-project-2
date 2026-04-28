import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
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
        layers,
        mo,
        models,
        os,
        pd,
        plt,
        preprocess_input,
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
    return


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
    return


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

    _Fully report your results, including the relevant predictive power, statistical significance, and/or validity of all implemented models. Discuss coefficients where relevant. Use tables and figures as needed._
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    _This section should include a fully developed interpretation that is consistent with the results and clearly addresses the research question. Discuss here any major caveats or limitations to the interpretation, the extent to which it can be generalized, and how it might be extended by further research._
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Uses of Python: Reflection

    _Take a step back and analyze your own use of code. Includes table of technical dependencies. Provide some rationale for choices you’ve made. Considerations may include performance, human readability, code dependencies, and reproducibility._
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    _List all works cited in the data guide. Use proper APA format._
    """)
    return


if __name__ == "__main__":
    app.run()
