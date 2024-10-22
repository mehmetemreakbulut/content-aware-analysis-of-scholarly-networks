import numpy as np

def pearson_correlation_rank(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    n = len(array1)

    # Convert scores to ranks
    rank1 = np.argsort(np.argsort(-np.array(array1))) + 1
    rank2 = np.argsort(np.argsort(-np.array(array2))) + 1

    # Handle ties by setting rank to average rank
    for ranks in [rank1, rank2]:
        unique_scores = np.unique(ranks)
        for score in unique_scores:
            mask = ranks == score
            if np.sum(mask) > 1:
                ranks[mask] = np.mean(np.arange(1, n+1)[mask])

    # Calculate average ranks
    R1_mean = np.mean(rank1)
    R2_mean = np.mean(rank2)

    # Calculate numerator and denominator
    numerator = np.sum((rank1 - R1_mean) * (rank2 - R2_mean))
    denominator = np.sqrt(np.sum((rank1 - R1_mean)**2) * np.sum((rank2 - R2_mean)**2))

    # Calculate correlation
    correlation = numerator / denominator

    return correlation

# Example usage
scores1 = [95, 90, 80, 85, 88, 92]
scores2 = [88, 92, 80, 85, 90, 95]

correlation = pearson_correlation_rank(scores1, scores2)
print(f"The correlation between the two rank lists is: {correlation}")
