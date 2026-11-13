# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import numpy as np
import tensorflow as tf

# calculate rank-1 accuracy, rank-5 accuracy, rank-10 accuracy
def calculate_rank_accuracy(distances, query_labels, gallery_labels, top_k=1):
    """
    Calcule la précision Rank-k (Rank-1, Rank-5, Rank-10) en TensorFlow.

    distances : tf.Tensor de forme [num_queries, num_gallery]
        Matrice des distances entre chaque query et chaque élément de la galerie.
    query_labels : tf.Tensor de forme [num_queries]
        Labels des queries.
    gallery_labels : tf.Tensor de forme [num_gallery]
        Labels de la galerie.
    top_k : int
        Nombre de premiers indices à considérer pour la précision Rank-k.

    Retourne un scalaire tf.Tensor représentant la précision Rank-k.
    """
    # Indices des distances triées par ordre croissant (plus proche = meilleure correspondance)
    sorted_indices = tf.argsort(distances, axis=1, direction='ASCENDING')

    # Sélection des top_k indices
    top_k_indices = sorted_indices[:, :top_k]  # shape: [num_queries, top_k]

    # Récupération des labels correspondants dans la galerie
    top_k_gallery_labels = tf.gather(gallery_labels, top_k_indices)  # shape: [num_queries, top_k]

    # Expansion des query_labels pour la comparaison
    query_labels_expanded = tf.expand_dims(query_labels, axis=1)  # shape: [num_queries, 1]

    # Vérification si le label query est dans les top_k labels gallery
    matches = tf.equal(top_k_gallery_labels, query_labels_expanded)  # shape: [num_queries, top_k]

    # Pour chaque query, on vérifie s'il y a au moins une correspondance dans top_k
    correct = tf.reduce_any(matches, axis=1)  # shape: [num_queries], bool

    # Calcul de la précision (nombre de corrects / nombre total de queries)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return accuracy.numpy()

# calculate the mAP (mean Average Precision)
def calculate_map(distances, query_labels, gallery_labels):
    """
    distances: tf.Tensor, shape (num_queries, num_gallery)
    query_labels: tf.Tensor, shape (num_queries,)
    gallery_labels: tf.Tensor, shape (num_gallery,)
    """
    # Get sorted indices of gallery samples per query (ascending distances)
    indices = tf.argsort(distances, axis=1, direction='ASCENDING')  # shape (num_queries, num_gallery)
    
    # Gather gallery labels sorted by distance for each query
    sorted_gallery_labels = tf.gather(gallery_labels, indices, batch_dims=0)  # shape (num_queries, num_gallery)
    
    # Expand query_labels to shape (num_queries, 1) for broadcasting
    query_labels_exp = tf.expand_dims(query_labels, axis=1)  # shape (num_queries, 1)
    
    # Create boolean mask where gallery label matches query label
    matches = tf.equal(sorted_gallery_labels, query_labels_exp)  # shape (num_queries, num_gallery), dtype=bool
    
    # Count number of relevant items per query
    relevant_counts = tf.reduce_sum(tf.cast(matches, tf.int32), axis=1)  # shape (num_queries,)
    
    # Avoid division by zero: mask queries with no relevant items
    nonzero_mask = relevant_counts > 0  # shape (num_queries,)
    
    # Compute cumulative sum of matches per query (to calculate precision at each position)
    matches_float = tf.cast(matches, tf.float32)
    cum_matches = tf.cumsum(matches_float, axis=1)  # shape (num_queries, num_gallery)
    
    # Create rank positions: 1, 2, ..., num_gallery
    rank_positions = tf.range(1, tf.shape(distances)[1] + 1, dtype=tf.float32)  # shape (num_gallery,)
    
    # Compute precision at each position: cum_matches / rank_positions
    precision_at_k = cum_matches / rank_positions  # shape (num_queries, num_gallery)
    
    # Mask precision values where matches==True (only consider relevant positions)
    precision_at_relevant = tf.where(matches, precision_at_k, tf.zeros_like(precision_at_k))  # zero where not relevant
    
    # Sum precision values at relevant positions per query
    sum_precisions = tf.reduce_sum(precision_at_relevant, axis=1)  # shape (num_queries,)
    
    # Calculate average precision per query: sum_precisions / relevant_counts
    average_precision = tf.where(nonzero_mask,
                                 sum_precisions / tf.cast(relevant_counts, tf.float32),
                                 tf.zeros_like(sum_precisions))
    
    # Calculate mean average precision over queries with at least one relevant item
    mAP = tf.reduce_sum(average_precision) / tf.reduce_sum(tf.cast(nonzero_mask, tf.float32))
    
    return mAP.numpy()

# calculate the correlation between the original and quantized features
def calculate_correlation(original_features, quantized_features):
    """
    Calcule la moyenne des corrélations de Pearson entre chaque paire de vecteurs
    dans original_features et quantized_features.

    original_features : tf.Tensor de forme [batch_size, feature_dim]
    quantized_features : tf.Tensor de même forme

    Retourne un scalaire tf.Tensor représentant la moyenne des corrélations.
    """
    # Calcul de la moyenne par ligne
    mean_orig = tf.reduce_mean(original_features, axis=1, keepdims=True)
    mean_quant = tf.reduce_mean(quantized_features, axis=1, keepdims=True)

    # Centrage des données
    orig_centered = original_features - mean_orig
    quant_centered = quantized_features - mean_quant

    # Calcul covariance (par ligne)
    covariance = tf.reduce_sum(orig_centered * quant_centered, axis=1)

    # Calcul des écarts-types (par ligne)
    std_orig = tf.sqrt(tf.reduce_sum(tf.square(orig_centered), axis=1))
    std_quant = tf.sqrt(tf.reduce_sum(tf.square(quant_centered), axis=1))

    # Calcul corrélation de Pearson (par ligne)
    correlation = covariance / (std_orig * std_quant + 1e-12)  # epsilon pour éviter division par 0

    # Moyenne des corrélations
    mean_correlation = tf.reduce_mean(correlation)

    return mean_correlation.numpy()

def pairwise_distance(embeddings1, embeddings2=None, distance_metric='euclidean', epsilon=1e-12):
    """
    Compute pairwise distances between two sets of embeddings.

    Args:
        embeddings1: Tensor of shape (N, D).
        embeddings2: Tensor of shape (M, D) or None. If None, embeddings2 = embeddings1.
        distance_metric: String, either 'euclidean' or 'cosine'.
        epsilon: Small float added for numerical stability in Euclidean distance.

    Returns:
        distances: Tensor of shape (N, M) with pairwise distances.
    """
    embeddings1 = tf.cast(embeddings1, tf.float32)
    if embeddings2 is not None:
        embeddings2 = tf.cast(embeddings2, tf.float32)
    else:
        embeddings2 = embeddings1

    # Normalize embeddings to unit length
    embeddings1 = tf.math.l2_normalize(embeddings1, axis=1)
    embeddings2 = tf.math.l2_normalize(embeddings2, axis=1)

    if distance_metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        distances = 1.0 - tf.matmul(embeddings1, embeddings2, transpose_b=True)
        # Clip to [0, 2] range to avoid small numerical errors outside valid cosine distance range
        distances = tf.clip_by_value(distances, 0.0, 2.0)

    elif distance_metric == 'euclidean':
        # embeddings shape: (N, D)
        squared_norms1 = tf.reduce_sum(tf.square(embeddings1), axis=1, keepdims=True)  # (N1, 1)
        squared_norms2 = tf.reduce_sum(tf.square(embeddings2), axis=1, keepdims=True)  # (N2, 1)

        # Compute inner product
        inner_prod = tf.matmul(embeddings1, embeddings2, transpose_b=True)  # (N1, N2)

        # Compute squared distances
        squared_dist = squared_norms1 - 2 * inner_prod + tf.transpose(squared_norms2)

        # Clamp and sqrt
        squared_dist = tf.maximum(squared_dist, 0.0)
        distances = tf.sqrt(squared_dist + epsilon)
    else:
        raise ValueError(f"Invalid distance_metric '{distance_metric}'. Choose from ['euclidean', 'cosine'].")

    return distances