import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd
from matplotlib.gridspec import GridSpec

def plot_rank_movement_heatmap(base_ranks, new_ranks, save_path=None):
    """
    Create a heatmap showing how papers move between different ranking brackets
    """
    # Create ranking brackets (e.g., top 10%, 20%, etc.)
    def get_bracket(rank, total):
        percentile = (rank / total) * 100
        return int(percentile // 10)

    total_papers = len(base_ranks)
    movements = np.zeros((10, 10))  # 10 brackets (0-10%, 10-20%, etc.)

    for paper in base_ranks.index:
        old_bracket = get_bracket(base_ranks[paper], total_papers)
        new_bracket = get_bracket(new_ranks[paper], total_papers)
        movements[old_bracket, new_bracket] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(movements, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Paper Movement Between Ranking Brackets')
    plt.xlabel('New Rank Bracket (Percentile)')
    plt.ylabel('Original Rank Bracket (Percentile)')

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_metric_contribution_analysis(base_scores, metric_scores, num_bins=10, save_path=None):
    """
    Analyze how the new metric contributes to final scores across different ranking levels
    """
    plt.figure(figsize=(15, 6))

    # Create bins based on base scores
    base_quantiles = pd.qcut(base_scores, num_bins, labels=False)

    # Calculate average metric contribution for each bin
    bin_contributions = pd.DataFrame({
        'base_score': base_scores,
        'metric_score': metric_scores,
        'bin': base_quantiles
    }).groupby('bin').agg({
        'base_score': 'mean',
        'metric_score': 'mean'
    })

    # Plot
    x = range(num_bins)
    width = 0.35

    plt.bar(x, bin_contributions['base_score'], width, label='Base Score', color='blue', alpha=0.6)
    plt.bar([i + width for i in x], bin_contributions['metric_score'], width, label='Topic Score', color='red', alpha=0.6)

    plt.xlabel('Paper Ranking Bracket (Lower = Better)')
    plt.ylabel('Average Score')
    plt.title('Score Composition Across Ranking Brackets')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_temporal_impact_analysis(years, base_ranks, new_ranks, save_path=None):
    """
    Analyze how the impact of the new metric varies across publication years
    """
    year_changes = {}
    for year in set(years.values()):
        year_papers = [p for p, y in years.items() if y == year]
        year_changes[year] = [base_ranks[p] - new_ranks[p] for p in year_papers]

    years_list = sorted(year_changes.keys())
    medians = [np.median(year_changes[y]) for y in years_list]
    quartile1 = [np.percentile(year_changes[y], 25) for y in years_list]
    quartile3 = [np.percentile(year_changes[y], 75) for y in years_list]

    plt.figure(figsize=(12, 6))
    plt.fill_between(years_list, quartile1, quartile3, alpha=0.3, color='blue')
    plt.plot(years_list, medians, 'b-', label='Median Change')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.title('Impact of Topic Score Across Publication Years')
    plt.xlabel('Publication Year')
    plt.ylabel('Rank Change')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_comprehensive_analysis(base_ranks, new_ranks, years, categories, save_path=None):
    """
    Create a comprehensive multi-plot analysis
    """
    plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2)

    # 1. Rank Changes Distribution
    plt.subplot(gs[0, 0])
    rank_changes = base_ranks - new_ranks
    plt.hist(rank_changes, bins=50, alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Distribution of Rank Changes')
    plt.xlabel('Rank Change')
    plt.ylabel('Count')

    # 2. Category Impact
    plt.subplot(gs[0, 1])
    category_changes = pd.DataFrame({
        'category': categories,
        'change': rank_changes
    }).groupby('category')['change'].agg(['mean', 'std']).reset_index()

    plt.bar(category_changes['category'],
            category_changes['mean'],
            yerr=category_changes['std'],
            capsize=5)
    plt.xticks(rotation=45)
    plt.title('Impact by Category')
    plt.ylabel('Average Rank Change')

    # 3. Temporal Analysis
    plt.subplot(gs[1, :])
    year_data = pd.DataFrame({
        'year': years,
        'change': rank_changes
    })
    sns.boxplot(x='year', y='change', data=year_data)
    plt.title('Rank Changes Across Years')
    plt.xlabel('Publication Year')
    plt.ylabel('Rank Change')

    # 4. Top Papers Movement
    plt.subplot(gs[2, :])
    top_100_base = set(base_ranks.nsmallest(100).index)
    top_100_new = set(new_ranks.nsmallest(100).index)

    stayed = len(top_100_base.intersection(top_100_new))
    only_base = len(top_100_base - top_100_new)
    only_new = len(top_100_new - top_100_base)

    plt.pie([stayed, only_base, only_new],
            labels=['Stayed in Top 100', 'Dropped from Top 100', 'Entered Top 100'],
            autopct='%1.1f%%',
            colors=['green', 'red', 'blue'])
    plt.title('Changes in Top 100 Papers')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_interactive_analysis(articles, base_ranks, new_ranks, years, categories):
    """
    Create an interactive analysis session
    """
    # Basic statistics
    print("Basic Statistics:")
    print(f"Total number of papers: {len(articles)}")
    print(f"Correlation between rankings: {spearmanr(base_ranks, new_ranks)[0]:.3f}")

    # Generate all visualizations
    plot_rank_movement_heatmap(base_ranks, new_ranks)
    plot_metric_contribution_analysis(base_ranks, new_ranks)
    plot_temporal_impact_analysis(years, base_ranks, new_ranks)
    plot_comprehensive_analysis(base_ranks, new_ranks, years, categories)

    # Additional analysis for top papers
    top_100_changes = pd.DataFrame({
        'paper_id': base_ranks.index,
        'base_rank': base_ranks,
        'new_rank': new_ranks,
        'rank_change': base_ranks - new_ranks,
        'year': [years[p] for p in base_ranks.index],
        'category': [categories[p] for p in base_ranks.index]
    }).nsmallest(100, 'base_rank')

    return top_100_changes
