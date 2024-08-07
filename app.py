from argparse import ArgumentParser

import numpy as np
import pandas as pd
from pynndescent import NNDescent
from sentence_transformers import SentenceTransformer

from fakegen import generate_sythentic_data

model = SentenceTransformer("all-MiniLM-L6-v2")


def recall_at_k(actual, predicted, k):
    """Calculates recall at k.

    Args:
      actual: List of actual relevant items.
      predicted: List of predicted items.
      k: Number of top predictions to consider.

    Returns:
      Recall at k.
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = len(actual_set.intersection(predicted_set))
    return intersection / len(actual_set)


def precision_at_k(actual, predicted, k):
    """Calculates precision at k.

    Args:
        actual: List of actual relevant items.
        predicted: List of predicted items.
        k: Number of top predictions to consider.

    Returns:
        Precision at k.
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = len(actual_set.intersection(predicted_set))
    return intersection / k


def generate_product_index(products_df: pd.DataFrame):
    """Generates product index using PyNNDescent.

    Args:
      products_df: Dataframe containing products info.

    Returns:
      Tuple of product index and product embeddings.
    """
    product_embeddings = model.encode(products_df["model_name"].tolist())
    product_index = NNDescent(product_embeddings, metric="cosine", n_neighbors=10)
    return product_index, product_embeddings


def create_user_embedding(
    user_id: int,
    purchase_history_df: pd.DataFrame,
    users_df: pd.DataFrame,
    product_embeddings: np.ndarray,
):
    """Creates user embedding/numerical representation by concatenating
    product and preference embeddings.

    Args:
      user_id: ID of the queried user.
      purchase_history_df: Dataframe containing purchase history info.
      users_df: Dataframe containing users info.
      product_embeddings: Numpy N-dimensional array.

    Returns:
      Concatenated user embedding.
    """
    user_purchases = purchase_history_df[purchase_history_df["user_id"] == user_id][
        "product_id"
    ].tolist()

    preference_embeddings = model.encode(
        users_df[users_df["user_id"] == user_id]["preferences"].tolist()
    ).reshape(-1)

    user_embedding = np.concatenate(
        [np.mean(product_embeddings[user_purchases], axis=0), preference_embeddings]
    )

    return user_embedding


def make_recommendations(
    user_id: int,
    users_df: pd.DataFrame,
    product_index: NNDescent,
    purchase_history_df: pd.DataFrame,
    product_embeddings: np.ndarray,
):
    """Predict similar products to user.

    Args:
      user_id: ID of the queried user.
      users_df: Dataframe containing user info.
      product_index: Generated index from the product embeddings.
      purchase_history_df: Dataframe containing purchase history info.
      product_embeddings: Numpy N-dimensional array.

    Returns:
      Index of recommendations.
    """
    top_n = 50
    user_embedding = create_user_embedding(
        user_id, purchase_history_df, users_df, product_embeddings
    )

    # Find similar products
    similar_products, distances = product_index.query(
        user_embedding.reshape(1, -1), k=top_n + 1
    )

    # Get products purchased
    similar_user_products = purchase_history_df[
        purchase_history_df["product_id"].isin(similar_products[0])
    ]["product_id"].value_counts()

    # Recommend top N products
    recommendations = similar_user_products.index[:top_n]

    return recommendations


def evaluate_model(recommendations: pd.Index, products_df: pd.DataFrame):
    """Evaluates model based on precision and recall score metrics.

    Args:
      recommendations: Index of recommendations.
      products_df: Dataframe containing products info.

    Returns:
      Precision and recall scores.
    """
    products = products_df[products_df.columns[0]].tolist()
    rec_list = recommendations.tolist()
    precision_score = precision_at_k(products, rec_list, 50)
    recall_score = recall_at_k(products, rec_list, 50)
    return precision_score, recall_score


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This is a recommendation engine that recommends products to users based on their purchase history and phone preferences."
    )
    parser.add_argument("user_id", choices=range(1, 1225), type=int)
    args = parser.parse_args()
    users_df, products_df, purchase_history_df = generate_sythentic_data()
    product_index, product_embeddings = generate_product_index(products_df)
    recommendations = make_recommendations(
        args.user_id, users_df, product_index, purchase_history_df, product_embeddings
    )

    model_name_mappings = products_df.set_index(products_df.columns[0])[
        "model_name"
    ].to_dict()
    textual_recommendations = [
        model_name_mappings[product_id] for product_id in recommendations
    ]
    print("Top 50 recommendations:")
    for product in textual_recommendations:
        print(f"- {product}")

    # Log model evaluation results
    precision, recall = evaluate_model(recommendations, products_df)
    print("\nEvaluation metrics:")
    print("Precision score: ", precision)
    print("Recall score: ", recall)
