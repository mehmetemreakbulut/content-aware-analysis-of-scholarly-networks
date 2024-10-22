import numpy as np
import math
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager

def calculate_pagerank(M):
    G = nx.DiGraph(M)
    page_rank = nx.pagerank(G, alpha=0.85)
    return page_rank



def article_article_weights(M, A_author, A_journal, A_topic, alpha=0.6, beta=0.3, gamma=0.1, lambda_value = 6):
    n_P = M.shape[0]
    semantic_weights = np.zeros((n_P, n_P))
    network_weights = np.zeros((n_P, n_P))
    neighbour_dict = {}
    author_dict = {}
    journal_dict = {}
    topic_dict = {}
    for i in range(n_P):
        print(i, n_P)
        i_cites = M[i, :]
        i_cited = M[:, i]
        # Neighbours of i is the set of articles that i cites and the articles that cite i
        cite_index = np.where(i_cites>0)[0]
        cited_index = np.where(i_cited>0)[0]
        #union of the two sets for neighbours
        neighbours_i = np.union1d(cite_index, cited_index)
        neighbour_dict[i] = neighbours_i

        authors = np.where(A_author[i, :] > 0)[0]
        author_dict[i] = authors

        journals = np.where(A_journal[i, :] > 0)[0][0]
        journal_dict[i] = journals

        topics = np.where(A_topic[i, :] > 0)[0]
        topic_dict[i] = topics

    for i in range(n_P):
        if i % 100 == 0:
            print(i, n_P)
        neighbours_i = neighbour_dict[i]
        author_i = author_dict[i]
        journal_i =journal_dict[i]
        topic_i = np.where(A_topic[i, :] > 0)[0]
        for j in range(n_P):
            neighbours_j = neighbour_dict[j]

            # Calculate the number of common neighbours between i and j
            number_common_neighbours = np.intersect1d(neighbours_i, neighbours_j).shape[0]

            denominator = math.sqrt(len(neighbours_i) * len(neighbours_j))


            ####AUTHOR

            # Calculate the number of common authors between i and j


            author_j = author_dict[j]

            number_common_author = np.intersect1d(author_i, author_j).shape[0]

            denominator_author = math.sqrt(len(author_i) * len(author_j))



            ####JOURNAL

            # Check if the journals of i and j are the same


            journal_j = journal_dict[j]

            same_journal = journal_i == journal_j


            ####TOPIC

            # Calculate the number of common topics between i and j


            topic_j =  topic_dict[j]

            number_common_topics = np.intersect1d(topic_i, topic_j).shape[0]

            denominator_topic = math.sqrt(len(topic_i) * len(topic_j))

            # Calculate the weight between i and j

            if denominator == 0:
                w_article = 0
            else:
                w_article = number_common_neighbours / denominator

            if denominator_author == 0:
                w_author = 0
            else:
                w_author = number_common_author / denominator_author

            w_journal = 0
            if same_journal:
                w_journal = 1

            network_similarity = alpha * w_article + beta * w_author + gamma * w_journal

            semantic_similarity = 0
            if denominator_topic != 0:
                semantic_similarity = number_common_topics / denominator_topic

            semantic_weights[i, j] = semantic_similarity
            network_weights[i, j] = network_similarity

    #find the median value of the semantic weights
    median_sem = np.median(semantic_weights)

    #find the median value of the network weights
    median_net = np.median(network_weights)

    for i in range(n_P):
        print(i, n_P)
        for j in range(n_P):
            a = (math.e) ** (lambda_value * (semantic_weights[i, j] - median_sem))
            b = (math.e) ** (lambda_value * (network_weights[i, j] - median_net))

            M[i, j] = a * semantic_weights[i, j] + b * network_weights[i, j]



    return M



def author_hub_score(paper_authority_scores, A_author, time_difference, author_papers):
    n_A = A_author.shape[1]
    hub_scores = np.zeros(n_A)
    for a in range(n_A):
        hub_score = 0
        related_papers = author_papers[str(a)]
        for p in related_papers:
            hub_score += paper_authority_scores[p] * (2**(time_difference[p][a]))

        hub_scores[a] = hub_score / len(related_papers)

    #normalize the hub scores to sum to 1
    hub_scores = hub_scores / hub_scores.sum()
    return hub_scores

def journal_hub_score(paper_authority_scores, A_journal, time_difference):
    n_J = A_journal.shape[1]
    hub_scores = np.zeros(n_J)
    for j in range(n_J):
        hub_score = 0
        related_papers = np.where(A_journal[:, j] > 0)[0]
        for p in related_papers:
            hub_score += paper_authority_scores[p] * (2**(time_difference[p][j]))

        hub_scores[j] = hub_score / len(related_papers)

    hub_scores = hub_scores / hub_scores.sum()
    return hub_scores


def topic_hub_score(paper_authority_scores, A_topic, time_difference, topic_papers):
    n_T = A_topic.shape[1]
    hub_scores = np.zeros(n_T)
    for t in range(n_T):
        hub_score = 0
        related_papers = topic_papers[str(t)] if str(t) in topic_papers else []
        if len(related_papers) == 0:
            hub_scores[t] = 0
            continue
        for p in related_papers:
            hub_score += paper_authority_scores[p] * (2**(time_difference[p][t]))

        hub_scores[t] = hub_score / len(related_papers)

    hub_scores = hub_scores / hub_scores.sum()
    return hub_scores

def article_hub_score(paper_authority_scores, M, article_time_difference, cites, cited):
    n_P = M.shape[0]
    hub_scores = np.zeros(n_P)

    for p in range(n_P):
        hub_score = 0
        try:
            p_cites = cites[str(p)]
        except:
            p_cites = []

        try:
            p_cited_by = cited[str(p)]
        except:
            p_cited_by = []

        for related_paper in p_cites:
            hub_score += paper_authority_scores[related_paper] * (2**(article_time_difference[p][related_paper]))

        for related_paper in p_cited_by:
            hub_score += paper_authority_scores[related_paper] * (2**(article_time_difference[related_paper][p]))

        hub_scores[p] = hub_score / (len(p_cites) + len(p_cited_by)) if len(p_cites) + len(p_cited_by) > 0 else 0

    hub_scores = hub_scores / hub_scores.sum()
    return hub_scores


def page_rank_article_i(i, M, authority_scores, cited, sum_of_papers_cited):
    '''
    i_cited_by = cited[str(i)]
    score = 0
    for j in i_cited_by:
        sum_of_papers_j_cites = M[j, :].sum() # which equals |outgoing links of j| in the original graph and paper
        score += (authority_scores[j] * M[j, i]) / sum_of_papers_j_cites
    '''
    i_cited_by = np.array(cited[str(i)])

    # Calculate weighted scores for all j citing i
    weighted_scores = authority_scores[i_cited_by] * M[i_cited_by, i]

    # Divide by sum_of_papers_cited and sum up
    return np.sum(weighted_scores / sum_of_papers_cited[i_cited_by])


def authority_score_from_authors_article_i(i, A_author, author_hub, time_difference, paper_authors):
    i_authors = paper_authors[str(i)]
    score = 0
    for a in i_authors:
        score += author_hub[a] * (1/(1+(1*time_difference[i][a])))
    # sum of scores transferred from all the authors to papers through author_hub scores
    Z_a = author_hub.sum()
    return score / Z_a

def authority_score_from_journals_article_i(i, A_journal, journal_hub, time_difference, paper_journals):
    i_journals = paper_journals[str(i)]
    score = 0
    for j in i_journals:
        score += journal_hub[j] * (1/(1+(1*time_difference[i][j])))
    # sum of scores transferred from all the journals to papers through journal_hub scores
    Z_j = journal_hub.sum()
    return score / Z_j

def authority_score_from_topics_article_i(i, A_topic, topic_hub, time_difference, paper_topics):
    i_topics =  paper_topics[str(i)]
    score = 0
    for t in i_topics:
        score += topic_hub[t] * (1/(1+(1*time_difference[i][t])))
    # sum of scores transferred from all the topics to papers through topic_hub scores
    Z_t = topic_hub.sum()
    return score / Z_t

def authority_score_from_articles_article_i(i, M, article_hub, article_time_difference, cited):
    i_cited_by = cited[str(i)]
    score = 0
    for j in i_cited_by:
        time_coef = 1/(1+(1*article_time_difference[j][i])) if article_time_difference[j][i] > 0 else 1
        score += article_hub[j] * M[j, i] * time_coef

    i_cites = np.where(M[i, :] > 0)[0]
    for j in i_cites:
        timecoef = 1/(1+(1*article_time_difference[i][j])) if article_time_difference[i][j] > 0 else 1
        score += article_hub[j] * M[i, j] * timecoef
    # sum of scores transferred from all the articles to papers through article_hub scores
    Z_p = article_hub.sum()
    return score / Z_p


T_CURRENT = 2024
import threading
import threading
from concurrent.futures import ThreadPoolExecutor

def process_article(i, M, A_author, A_journal, A_topic, article_authority_scores, time_difference, author_papers, topic_papers, article_time_difference, cites, cited, time_vector, paper_authors, paper_journals, paper_topics, sum_of_papers_cited, alpha, beta, gamma, delta, omega, sigma, n_P):
    #update the authority score of articles
    page_rank_i = page_rank_article_i(i, M, article_authority_scores, cited, sum_of_papers_cited)
    #hub scores
    author_hub = author_hub_score(article_authority_scores, A_author, time_difference, author_papers)
    journal_hub = journal_hub_score(article_authority_scores, A_journal, time_difference)
    topic_hub = topic_hub_score(article_authority_scores, A_topic, time_difference, topic_papers)
    article_hub = article_hub_score(article_authority_scores, M, article_time_difference, cites, cited)

    #authority scores
    authority_from_authors = authority_score_from_authors_article_i(i, A_author, author_hub, time_difference, paper_authors)

    authority_from_journals = authority_score_from_journals_article_i(i, A_journal, journal_hub, time_difference, paper_journals)
    authority_from_topics = authority_score_from_topics_article_i(i, A_topic, topic_hub, time_difference, paper_topics)
    authority_from_articles = authority_score_from_articles_article_i(i, M, article_hub, article_time_difference, cited)

    #update the authority score of article i
    new_score = alpha * page_rank_i + beta * authority_from_authors + gamma * authority_from_journals + delta * authority_from_topics + omega * authority_from_articles + sigma * time_vector[i] + (1 - alpha - beta - gamma - delta - omega - sigma) * (1/n_P)

    print("Article ", i, " finished")
    return i, new_score



import concurrent.futures
def article_authority_score(M, A_author, A_journal, A_topic, time_difference, article_time_difference, article_years, cites, cited, author_papers, topic_papers, paper_authors, paper_journals, paper_topics,time_p = 0.62, alpha = 0.3, beta = 0.2, gamma = 0.1, delta=0.1, omega = 0.2, sigma = 0.1, epsilon = 0.0001):

    time_vector = np.ones(M.shape[0])
    for i in range(len(time_vector)):
        time_vector[i] = math.e ** (-time_p * (T_CURRENT - article_years[i]))

    #initialize the authority scores of articles to 1 / n_P
    n_P = M.shape[0]
    article_authority_scores = np.ones(n_P) / n_P

    #while not converged
    n = 0
    while True:
        print("Iteration: ", 32)
        old_scores = article_authority_scores.copy()
        results = [None] * n_P  # Initialize with None to keep track of completion

        with ThreadPoolExecutor(max_workers=32) as executor:
            # Submit all tasks to the executo
            sum_of_papers_cited = M.sum(axis=1)
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


        article_authority_scores = np.array(results)
        #normalize the authority scores
        article_authority_scores = article_authority_scores / article_authority_scores.sum()

        #check for convergence
        if np.linalg.norm(article_authority_scores - old_scores) < epsilon:
            break

        n += 1

    return article_authority_scores
