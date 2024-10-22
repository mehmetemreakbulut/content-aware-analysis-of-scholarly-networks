import numpy as np
import math
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager

def calculate_pagerank(M):
    G = nx.DiGraph(M)
    page_rank = nx.pagerank(G, alpha=0.85)
    return page_rank


from scipy.sparse import csr_matrix

def article_article_weights(M, A_author, A_journal, A_topic, BASE_WEIGHT = 0.5, alpha=0.3, beta=0.15, gamma=0.05, lambda_value = 6):
    n_P = M.shape[0]

    # Convert input matrices to sparse format for efficiency
    M_sparse = csr_matrix(M)
    A_author_sparse = csr_matrix(A_author)
    A_topic_sparse = csr_matrix(A_topic)

    # Precompute neighbors, authors, journals, and topics
    neighbor_dict = {i: set(M_sparse[i].nonzero()[1]).union(M_sparse[:, i].nonzero()[0]) for i in range(n_P)}
    author_dict = {i: set(A_author_sparse[i].nonzero()[1]) for i in range(n_P)}
    journal_dict = {i: A_journal[i].nonzero()[0][0] if A_journal[i].any() else -1 for i in range(n_P)}
    topic_dict = {i: set(A_topic_sparse[i].nonzero()[1]) for i in range(n_P)}

    # Precompute lengths
    neighbor_lengths = {i: len(neighbors) for i, neighbors in neighbor_dict.items()}
    author_lengths = {i: len(authors) for i, authors in author_dict.items()}
    topic_lengths = {i: len(topics) for i, topics in topic_dict.items()}

    semantic_weights = np.zeros((n_P, n_P))
    network_weights = np.zeros((n_P, n_P))

    for i in range(n_P):
        print(i)
        neighbors_i = neighbor_dict[i]
        authors_i = author_dict[i]
        journal_i = journal_dict[i]
        topics_i = topic_dict[i]

        for j in range(i+1, n_P):  # We only need to compute upper triangle
            neighbors_j = neighbor_dict[j]
            authors_j = author_dict[j]
            journal_j = journal_dict[j]
            topics_j = topic_dict[j]

            # Network similarity calculations
            common_neighbors = len(neighbors_i.intersection(neighbors_j))
            denominator = math.sqrt(neighbor_lengths[i] * neighbor_lengths[j])
            w_article = common_neighbors / denominator if denominator else 0

            common_authors = len(authors_i.intersection(authors_j))
            denominator_author = math.sqrt(author_lengths[i] * author_lengths[j])
            w_author = common_authors / denominator_author if denominator_author else 0

            w_journal = int(journal_i == journal_j)

            base = BASE_WEIGHT if M[i, j] != 0 else 0
            network_similarity = base + alpha * w_article + beta * w_author + gamma * w_journal

            # Semantic similarity calculation
            common_topics = len(topics_i.intersection(topics_j))
            denominator_topic = math.sqrt(topic_lengths[i] * topic_lengths[j])
            semantic_similarity = common_topics / denominator_topic if denominator_topic else 0

            # Store the results (symmetrically)
            semantic_weights[i, j] = semantic_weights[j, i] = semantic_similarity
            network_weights[i, j] = network_weights[j, i] = network_similarity

    #find the median value of the semantic weights
    median_sem = np.median(semantic_weights)

    #find the median value of the network weights
    median_net = np.median(network_weights)

    for i in range(n_P):
        for j in range(n_P):
            a = (math.e) ** (lambda_value * (semantic_weights[i, j] - median_sem))
            b = (math.e) ** (lambda_value * (network_weights[i, j] - median_net))

            M[i, j] = a * semantic_weights[i, j] + b * network_weights[i, j]



    return M



def author_hub_score(paper_authority_scores, A_author, time_diff_exp, author_papers):

    n_A = A_author.shape[1]

    # Convert author_papers to a list of arrays for efficient indexing
    author_papers_list = [np.array(author_papers[str(a)], dtype=int) for a in range(n_A)]

    # Calculate 2^(time_difference) for all papers and authors


    # Calculate hub scores for all authors
    hub_scores = np.array([
        np.sum(paper_authority_scores[papers] * time_diff_exp[papers, a]) / len(papers)
        for a, papers in enumerate(author_papers_list)
    ])

    # Normalize the hub scores to sum to 1
    hub_scores /= np.sum(hub_scores)

    return hub_scores

import numpy as np

def journal_hub_score(paper_authority_scores, A_journal, time_diff_exp, journal_papers):
    n_J = A_journal.shape[1]

    # Pre-calculate 2^(time_difference)


    # Calculate hub scores for all journals
    related_papers = [np.array(journal_papers[str(j)], dtype=int) for j in range(n_J)]

    hub_scores = np.array([
        np.sum(paper_authority_scores[papers] * time_diff_exp[papers, j]) / len(papers)
        if len(papers) > 0 else 0
        for j, papers in enumerate(related_papers)
    ])

    # Normalize the hub scores
    hub_scores /= np.sum(hub_scores)

    return hub_scores


import numpy as np

def topic_hub_score(paper_authority_scores, A_topic, time_diff_exp, topic_papers):
    n_T = A_topic.shape[1]

    # Pre-calculate 2^(time_difference)


    # Calculate hub scores for all topics
    hub_scores = np.zeros(n_T)
    for t in range(n_T):
        related_papers = np.array(topic_papers.get(str(t), []), dtype=int)
        if len(related_papers) > 0:
            hub_scores[t] = np.sum(paper_authority_scores[related_papers] * time_diff_exp[related_papers, t]) / len(related_papers)

    # Normalize the hub scores
    hub_scores /= np.sum(hub_scores)

    return hub_scores

import numpy as np

def article_hub_score(paper_authority_scores, M, time_diff_exp, cites, cited):
    n_P = M.shape[0]

    # Pre-calculate 2^(article_time_difference)


    # Calculate hub scores for all articles
    hub_scores = np.zeros(n_P)
    for p in range(n_P):
        p_cites = np.array(cites.get(str(p), []), dtype=int)
        p_cited_by = np.array(cited.get(str(p), []), dtype=int)

        score_cites = np.sum(paper_authority_scores[p_cites] * time_diff_exp[p, p_cites]) if len(p_cites) > 0 else 0
        score_cited_by = np.sum(paper_authority_scores[p_cited_by] * time_diff_exp[p_cited_by, p]) if len(p_cited_by) > 0 else 0

        total_related = len(p_cites) + len(p_cited_by)
        hub_scores[p] = (score_cites + score_cited_by) / total_related if total_related > 0 else 0

    # Normalize the hub scores
    hub_scores /= np.sum(hub_scores)

    return hub_scores


def page_rank_article_i(i, M, authority_scores, cited, sum_of_papers_cited):
    '''
    i_cited_by = cited[str(i)]
    score = 0
    for j in i_cited_by:
        sum_of_papers_j_cites = M[j, :].sum() # which equals |outgoing links of j| in the original graph and paper
        score += (authority_scores[j] * M[j, i]) / sum_of_papers_j_cites
    '''
    i_cited_by = np.array(cited[str(i)]) if str(i) in cited else np.array([])

    # Calculate weighted scores for all j citing i
    weighted_scores = 0
    for j in i_cited_by:
        a = authority_scores[j]
        b = M[j, i]
        sum_of_papers_j_cites = sum_of_papers_cited[j]
        weighted_scores += (a * b) / sum_of_papers_j_cites if sum_of_papers_j_cites > 0 else 0

    return weighted_scores


def authority_score_from_authors_article(A_author, author_hub, time_difference, paper_authors):
    n_papers = A_author.shape[0]

    scores = np.zeros(n_papers)

    Z_a = np.sum(author_hub)

    for i in range(n_papers):
        i_authors = np.array(paper_authors[str(i)], dtype=int)
        scores[i] = np.sum(author_hub[i_authors] * (1/(1+time_difference[i, i_authors])))

        scores[i] = scores[i] / Z_a

    return scores / Z_a

def authority_score_from_journals_article(A_journal, journal_hub, time_difference, paper_journals):
    n_papers = A_journal.shape[0]
    scores = np.zeros(n_papers)


    # Normalize scores
    Z_j = np.sum(journal_hub)
    for i in range(n_papers):
        i_journals = np.array(paper_journals[str(i)], dtype=int)
        scores[i] = np.sum(journal_hub[i_journals] * (1/(1+time_difference[i, i_journals])))

        scores[i] = scores[i] / Z_j

    return scores

def authority_score_from_topics_article(A_topic, topic_hub, time_difference, paper_topics):
    n_papers = A_topic.shape[0]
    scores = np.zeros(n_papers)
    # Normalize scores
    Z_t = np.sum(topic_hub)


    for i in range(n_papers):
        i_topics = np.array(paper_topics[str(i)], dtype=int)
        scores[i] = np.sum(topic_hub[i_topics] * (1/(1+time_difference[i, i_topics])))

        scores[i] = scores[i] / Z_t

    return scores

def authority_score_from_articles_article(M, article_hub, time_diff_frac_article, cites, cited):
    n_papers = M.shape[0]
    scores = np.zeros(n_papers)

    # Normalize scores
    Z_p = np.sum(article_hub)
    for i in range(n_papers):
        # Cited by
        i_cited_by = np.array(cited.get(str(i), []), dtype=int)
        if len(i_cited_by) > 0:
            scores[i] += np.sum(article_hub[i_cited_by] * M[i_cited_by, i] * time_diff_frac_article[i_cited_by, i])

        # Cites
        i_cites = np.array(cites.get(str(i), []), dtype=int)
        if len(i_cites) > 0:
            scores[i] += np.sum(article_hub[i_cites] * M[i, i_cites] * time_diff_frac_article[i, i_cites])

        scores[i] = scores[i] / Z_p

    return scores
T_CURRENT = 2024
import threading
import threading
from concurrent.futures import ThreadPoolExecutor

def process_article(M, A_author, A_journal, A_topic, article_authority_scores, time_difference, time_diff_exp, author_papers, topic_papers, journal_papers, article_time_difference,time_diff_exp_article, time_diff_frac_article, cites, cited, time_vector, paper_authors, paper_journals, paper_topics, sum_of_papers_cited, alpha, beta, gamma, delta, omega, sigma, n_P):
    #update the authority score of articles
    print(f"Processing article")
    page_rank_scores = np.zeros(n_P)
    for i in range(n_P):
        page_rank_scores[i] = page_rank_article_i(i, M, article_authority_scores, cited, sum_of_papers_cited)
    print(f"Page rank scores: {page_rank_scores}")
    #hub scores,
    author_hub = author_hub_score(article_authority_scores, A_author, time_diff_exp, author_papers)
    print(f"Author hub: {author_hub}")
    journal_hub = journal_hub_score(article_authority_scores, A_journal, time_diff_exp, journal_papers)
    print(f"Journal hub: {journal_hub}")
    topic_hub = topic_hub_score(article_authority_scores, A_topic, time_diff_exp, topic_papers)

    print(f"Calculating hub scores for article")
    article_hub = article_hub_score(article_authority_scores, M, time_diff_exp_article, cites, cited)

    print(f"Calculating authority scores for article")
    #authority scores
    authority_scores = np.zeros(n_P)

    authority_from_authors = authority_score_from_authors_article(A_author, author_hub, time_difference, paper_authors)
    print(f"Authority from authors: {authority_from_authors}")
    authority_from_journals = authority_score_from_journals_article(A_journal, journal_hub, time_difference, paper_journals)
    print(f"Authority from journals: {authority_from_journals}")
    authority_from_topics = authority_score_from_topics_article(A_topic, topic_hub, time_difference, paper_topics)
    print(f"Authority from topics: {authority_from_topics}")
    authority_from_articles = authority_score_from_articles_article(M, article_hub, time_diff_frac_article, cites, cited)
    print(f"Authority from articles: {authority_from_articles}")
    for i in range(n_P):
        #update the authority score of article i
        new_score = alpha * page_rank_scores[i] + beta * authority_from_authors[i] + gamma * authority_from_journals[i] + delta * authority_from_topics[i] + omega * authority_from_articles[i] + sigma * time_vector[i] + (1 - alpha - beta - gamma - delta - omega - sigma) * (1/n_P)
        authority_scores[i] = new_score
    return authority_scores



import concurrent.futures
import time


def csr_allclose(a, b, atol = 0.0001):
    c = a - b
    #see max absolute difference
    print(f"Max absolute difference: {np.max(np.abs(c.data))}")
    return np.allclose(c.data, 0, atol=atol)


def article_authority_score(M, A_author, A_journal, A_topic, time_difference, article_time_difference, article_years, cites, cited, author_papers, topic_papers, journal_papers, paper_authors, paper_journals, paper_topics,time_p = 0.62, alpha = 0.3, beta = 0.2, gamma = 0.1, delta=0.1, omega = 0.2, sigma = 0.1, epsilon = 0.0001):

    time_vector = np.ones(M.shape[0])
    for i in range(len(time_vector)):
        time_vector[i] = math.e ** (-time_p * (T_CURRENT - article_years[i]))

    #normalize time vector
    time_vector = time_vector / np.sum(time_vector)
    #initialize the authority scores of articles to 1 / n_P
    n_P = M.shape[0]
    article_authority_scores = np.ones(n_P) / n_P

    #while not converged
    n = 0
    sum_of_papers_cited = np.zeros(n_P)
    for i in range(n_P):
        #the sum of weights from paper i to all papers it cites
        if str(i) in cites:
            for j in cites[str(i)]:
                sum_of_papers_cited[i] += M[i, j]
        else:
            sum_of_papers_cited[i] = 0


    print(f"Sum of papers cited: {sum_of_papers_cited}")
    #exp time
    time_diff_exp = np.exp2(time_difference)

    time_diff_exp_article = np.exp2(article_time_difference)

    # Pre-calculate time coefficients
    time_diff_frac_article = 1 / (1 + article_time_difference)
    time_diff_frac_article[article_time_difference <= 0] = 1


    while True:
        print("Iteration: ", n)
        old_scores = article_authority_scores.copy()
        """
        results = [None] * n_P  # Initialize with None to keep track of completion

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit all tasks to the executo

            futures = [executor.submit(process_article, i, M, A_author, A_journal, A_topic, article_authority_scores,
                                       time_difference, author_papers, topic_papers, article_time_difference,
                                       cites, cited, time_vector, paper_authors, paper_journals, paper_topics, sum_of_papers_cited, alpha, beta, gamma, delta, omega, sigma, n_P) for i in range(n_P)]

            # Retrieve results as they complete
            for future in futures:
                i, new_score = future.result()
                results[i] = new_score


        # Check if all results are computed
        if None in results:
            raise RuntimeError("Some articles were not processed successfully")
        """

        stime = time.time()
        new_scores = process_article(M, A_author, A_journal, A_topic, article_authority_scores,
                                       time_difference,time_diff_exp, author_papers, topic_papers, journal_papers, article_time_difference, time_diff_exp_article, time_diff_frac_article,
                                       cites, cited, time_vector, paper_authors, paper_journals, paper_topics, sum_of_papers_cited, alpha, beta, gamma, delta, omega, sigma, n_P)
        article_authority_scores = np.array(new_scores)

        #normalize the authority scores
        article_authority_scores = article_authority_scores / article_authority_scores.sum()
        etime = time.time()
        print("seconds", (etime-stime))
        #check for convergence
        print("Old scores: ", old_scores)
        print("New scores: ", article_authority_scores)

        if csr_allclose(old_scores, article_authority_scores, epsilon):
            break

        n += 1

    return article_authority_scores
