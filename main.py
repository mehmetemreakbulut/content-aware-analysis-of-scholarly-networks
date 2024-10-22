import datetime
import json
from math import sqrt
import math
import numpy as np
import pandas as pd
from utils import calculate_pagerank
import med_rank
import utils
import utils

papers = json.load(open('../dataset_papers.json', 'r'))
citations = pd.read_csv('../dataset_citations.csv')


try:
    papers_index = json.load(open('useful/papers_index.json', 'r'))
    index_citations = json.load(open('useful/index_citations.json', 'r'))
    index_references = json.load(open('useful/index_references.json', 'r'))
except:
    papers_index = {}
    index_citations = {}
    index_references = {}
    for i, paper in enumerate(papers):
        papers_index[paper['paperId']] = i
        index_citations[i] = 0
        index_references[i] = 0

    json.dump(papers_index, open('useful/papers_index.json', 'w'))
    json.dump(index_citations, open('useful/index_citations.json', 'w'))
    json.dump(index_references, open('useful/index_references.json', 'w'))



try:
    journals = json.load(open('useful/journals.json', 'r'))
except:
    journals = []
    for paper in papers:
        if paper['journal'] not in journals:
            journals.append(paper['journal'])
journals = list(journals)

try:
    jounals_index = json.load(open('useful/journals_index.json', 'r'))
except:
    journals_index = {}
    for i, journal in enumerate(journals):
        journals_index[journal] = i

    json.dump(journals_index, open('useful/journals_index.json', 'w'))

try:
    authors = json.load(open('useful/authors.json', 'r'))
except:
    authors = []
    for paper in papers:
        for author in paper['authors']:
            if author not in authors:
                authors.append(author)

    json.dump(authors, open('useful/authors.json', 'w'))
authors = list(authors)

try:
    authors_index = json.load(open('useful/authors_index.json', 'r'))
except:
    authors_index = {}
    for i, author in enumerate(authors):
        authors_index[author] = i

    json.dump(authors_index, open('useful/authors_index.json', 'w'))


#if citation_adj_matrix file exists, load it
try:
    citations_adj_matrix = np.load('matrix/citations_adj_matrix.npy')
    print("citations_adj_matrix loaded")
except:
    print("citations_adj_matrix not found")
    citations_adj_matrix = np.zeros((len(papers), len(papers)))
    for i, row in citations.iterrows():
        source = row['source']
        target = row['target']
        source_index = papers_index[source]
        target_index = papers_index[target]
        citations_adj_matrix[source_index][target_index] = 1
        index_citations[target_index] += 1
        index_references[source_index] += 1


    np.save('matrix/citations_adj_matrix.npy', citations_adj_matrix)

print("citations_adj_matrix")

try:
    paper_author_adj_matrix = np.load('matrix/paper_author_adj_matrix.npy')
    print("paper_author_adj_matrix loaded")
except:
    print("paper_author_adj_matrix not found")
    paper_author_adj_matrix = np.zeros((len(papers), len(authors)))
    for i, paper in enumerate(papers):
        for author in paper['authors']:
            author_index = authors_index[author]
            paper_author_adj_matrix[i][author_index] = 1
    np.save('matrix/paper_author_adj_matrix.npy', paper_author_adj_matrix)

print("paper_author_adj_matrix")

try:
    paper_journal_adj_matrix = np.load('matrix/paper_journal_adj_matrix.npy')
    print("paper_journal_adj_matrix loaded")
except:
    print("paper_journal_adj_matrix not found")
    paper_journal_adj_matrix = np.zeros((len(papers), len(journals)))
    for i, paper in enumerate(papers):
        journal_index = journals_index[paper['journal']]
        paper_journal_adj_matrix[i][journal_index] = 1

    np.save('matrix/paper_journal_adj_matrix.npy', paper_journal_adj_matrix)

print("paper_journal_adj_matrix")





type_ids = []
type_file = open('../type_ids.txt', 'r')
lines = type_file.readlines()
for line in lines:
    type = line.split('|')[1]
    if type in ["T052","T185", "T077", "T060", "T056" ,"T065", "T050", "T071", "T051", "T099", "T169", "T064", "T058", "T078", "T170", "T063", "T066", "T041", "T070", "T057", "T090", "T067", "T098", "T097", "T094", "T080", "T081", "T062", "T089", "T167", "T095", "T054", "T082", "T079"]:
        continue
    type_ids.append(type)
type_ids = set(type_ids)
type_file.close()





entities = json.load(open('../paper_with_annotations.json', 'r'))


#open cuis.txt
cuis = []
cuis_file = open('../cuis.txt', 'r')
lines = cuis_file.readlines()
for line in lines:
    cuis.append(line.strip())

cuis_file.close()

if len(cuis) == 0:
    for key, value in entities.items():
        for key2, value2 in value["entities"].items():
            cui = value2["cui"]
            acc = value2["acc"]
            if acc < 0.75:
                continue
            tids = value2["type_ids"]
            not_valid = False
            for type_id in tids:
                if type_id not in type_ids:
                    not_valid = True
                    break
            if not not_valid:
                if cui not in cuis:
                    cuis.append(cui)

    cuis_file = open('../cuis.txt', 'w')
    for cui in cuis:
        cuis_file.write(cui + "\n")
    cuis_file.close()


cuis_index = {}
for i, cui in enumerate(cuis):
    cuis_index[cui] = i

try:
    cuis_adj_matrix = np.load('matrix/cuis_adj_matrix.npy')
    print("cuis_adj_matrix loaded")

except:
    cuis_adj_matrix = np.zeros((len(papers), len(cuis)))
    for i, paper in enumerate(papers):
        for key, value in entities[paper['paperId']]["entities"].items():
            cui = value["cui"]
            acc = value["acc"]
            if cui in cuis_index and acc > 0.75:
                cui_index = cuis_index[cui]
                cuis_adj_matrix[i][cui_index] = 1

    np.save('matrix/cuis_adj_matrix.npy', cuis_adj_matrix)

print("cuis_adj_matrix")



try:
    article_time_difference = np.load('matrix/article_time_difference.npy')
    print("article_time_difference loaded")
except:
    article_time_difference = np.zeros((len(papers), len(papers)))
    for i, row in citations.iterrows():
        source = row['source']
        target = row['target']
        source_index = papers_index[source]
        target_index = papers_index[target]
        source_time = papers[source_index]['publicationDate']
        source_time = int(source_time.split('-')[0])
        target_time = papers[target_index]['publicationDate']
        target_time = int(target_time.split('-')[0])
        article_time_difference[source_index][target_index] = abs(source_time - target_time)

    np.save('matrix/article_time_difference.npy', article_time_difference)

print("article_time_difference")


T_CURRENT = 2024
try:
    time_difference = np.load('matrix/time_difference.npy')
    print("time_difference loaded")
except:
    time_difference = np.zeros((len(papers), len(authors)))
    for i, paper in enumerate(papers):
        for author in paper['authors']:
            author_index = authors_index[author]
            year = paper['publicationDate']
            year = int(year.split('-')[0])
            time_difference[i][author_index] = T_CURRENT - year

    np.save('matrix/time_difference.npy', time_difference)

print("time_difference")


try:
    print("citations_adj_matrix will be loaded")
    citation_weight_adj_matrix = np.load('matrix/citations_weighted_adj_matrix.npy')
    print("citation_weight_matrix loaded")

except:
    citation_weight_adj_matrix = utils.article_article_weights(citations_adj_matrix, paper_author_adj_matrix, paper_journal_adj_matrix, cuis_adj_matrix)
    np.save('matrix/citations_weighted_adj_matrix.npy', citation_weight_adj_matrix)

print("citation_weight_matrix")

article_years = {}
for i, paper in enumerate(papers):
    article_years[i] = int(paper['publicationDate'].split('-')[0])


try:
    cites = json.load(open('matrix/cites.json', 'r'))
    print("cites loaded")
except:
    print("cites not found")
    cites = {}
    for i, row in citations.iterrows():
        source = row['source']
        target = row['target']
        source_index = papers_index[source]
        target_index = papers_index[target]
        if source_index not in cites:
            cites[source_index] = []
        cites[source_index].append(target_index)

    with open('matrix/cites.json', 'w') as f:
        json.dump(cites, f)
try:
    cited = json.load(open('matrix/cited.json', 'r'))
    print("cited loaded")
except:
    print("cited not found")
    cited = {}
    for i, row in citations.iterrows():
        source = row['source']
        target = row['target']
        source_index = papers_index[source]
        target_index = papers_index[target]
        if target_index not in cited:
            cited[target_index] = []
        cited[target_index].append(source_index)

    with open('matrix/cited.json', 'w') as f:
        json.dump(cited, f)


author_papers = {}
try:
    author_papers = json.load(open('matrix/author_papers.json', 'r'))
    print("author_papers loaded")
except:
    print("author_papers not found")
    author_papers = {}
    #iterate on paper_author_adj_matrix
    for i, row in enumerate(paper_author_adj_matrix):
        for j, value in enumerate(row):
            if value > 0:
                if j not in author_papers:
                    author_papers[j] = []
                author_papers[j].append(i)

    with open('matrix/author_papers.json', 'w') as f:
        json.dump(author_papers, f)

print("author_papers")



topic_papers = {}
try:
    topic_papers = json.load(open('matrix/topic_papers.json', 'r'))
    print("topic_papers loaded")
except:
    print("topic_papers not found")
    topic_papers = {}
    #iterate on paper_author_adj_matrix
    for i, row in enumerate(cuis_adj_matrix):
        for j, value in enumerate(row):
            if value > 0:
                if j not in topic_papers:
                    topic_papers[j] = []
                topic_papers[j].append(i)

    with open('matrix/topic_papers.json', 'w') as f:
        json.dump(topic_papers, f)


journal_papers  = {}
try:
    journal_papers = json.load(open('matrix/journal_papers.json', 'r'))
    print("journal_papers loaded")
except:
    print("journal_papers not found")
    journal_papers = {}
    #iterate on paper_author_adj_matrix
    for i, row in enumerate(paper_journal_adj_matrix):
        for j, value in enumerate(row):
            if value > 0:
                if j not in journal_papers:
                    journal_papers[j] = []
                journal_papers[j].append(i)

    with open('matrix/journal_papers.json', 'w') as f:
        json.dump(journal_papers, f)

paper_authors = {}
try:
    paper_authors = json.load(open('matrix/paper_authors.json', 'r'))
    print("paper_authors loaded")
except:
    paper_authors = {}
    for i, row in enumerate(paper_author_adj_matrix):
        if i not in paper_authors:
            paper_authors[i] = []
        for j, value in enumerate(row):
            if value > 0:
                paper_authors[i].append(j)

    with open('matrix/paper_authors.json', 'w') as f:
        json.dump(paper_authors, f)

paper_journals = {}
try:
    paper_journals = json.load(open('matrix/paper_journals.json', 'r'))
    print("paper_journals loaded")
except:
    paper_journals = {}
    for i, row in enumerate(paper_journal_adj_matrix):
        if i not in paper_journals:
            paper_journals[i] = []
        for j, value in enumerate(row):
            if value > 0:
                paper_journals[i].append(j)

    with open('matrix/paper_journals.json', 'w') as f:
        json.dump(paper_journals, f)


paper_topics = {}
try:
    paper_topics = json.load(open('matrix/paper_topics.json', 'r'))
    print("paper_topics loaded")
except:
    print("paper_topics not found")
    paper_topics = {}
    for i, row in enumerate(cuis_adj_matrix):
        if i not in paper_topics:
            paper_topics[i] = []
        for j, value in enumerate(row):
            if value > 0:
                paper_topics[i].append(j)

    with open('matrix/paper_topics.json', 'w') as f:
        json.dump(paper_topics, f)

'''
results = utils.article_authority_score(citation_weight_adj_matrix, paper_author_adj_matrix, paper_journal_adj_matrix, cuis_adj_matrix, time_difference, article_time_difference, article_years, cites, cited, author_papers, topic_papers, journal_papers, paper_authors, paper_journals, paper_topics,
    alpha=0.0, beta=0.3, gamma = 0.3, delta = 0.1, omega=0.3, sigma=0.0)

#save results

np.save('results/run17.npy', results)

print("results saved")
'''

from dataclasses import dataclass
@dataclass
class metric:
    alpha: float
    beta: float
    gamma: float
    delta: float
    omega: float
    sigma: float


no_topic_run = metric(alpha = 0.1, beta = 0.2, gamma = 0.2, delta=0.0, omega = 0.2, sigma = 0.1)
topicLOW_run = metric( alpha=0.1, beta=0.2, gamma = 0.2, delta = 0.1, omega=0.2, sigma=0.1)
topicMID_run = metric( alpha=0.1, beta=0.1, gamma = 0.1, delta = 0.4, omega=0.1, sigma=0.1)
topicHIGH_run= metric( alpha=0.1, beta=0.0, gamma = 0.0, delta = 0.7, omega=0.0, sigma=0.1)


no_topic_alphaMID_run = metric(alpha = 0.4, beta = 0.1, gamma = 0.1, delta=0.0, omega = 0.2, sigma = 0.1 )
topicLOW_alphaMID_run= metric( alpha=0.4, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.1, sigma=0.1 )
topicMID_alphaMID_run= metric(alpha=0.4, beta=0.0, gamma = 0.0, delta = 0.4, omega=0.0, sigma=0.1 )

no_topic_alphaHIGH_run = metric( alpha = 0.7, beta = 0.03, gamma = 0.03, delta=0.0, omega = 0.03, sigma = 0.1)
topicLOW_alphaHIGH_run= metric( alpha=0.7, beta=0.00, gamma = 0.00, delta = 0.1, omega=0.0, sigma=0.1 )


topic_and_author = metric( alpha = 0.1, beta = 0.3, gamma = 0.0, delta=0.4, omega = 0.0, sigma = 0.1)
topic_and_journal= metric( alpha = 0.1, beta = 0.0, gamma = 0.3, delta=0.4, omega = 0.0, sigma = 0.1 )
topic_and_article   = metric( alpha = 0.1, beta = 0.0, gamma = 0.0, delta=0.4, omega = 0.3, sigma = 0.1)


evaluation = [no_topic_run, topicLOW_run, topicMID_run, topicHIGH_run, no_topic_alphaMID_run, topicLOW_alphaMID_run, topicMID_alphaMID_run, no_topic_alphaHIGH_run, topicLOW_alphaHIGH_run, topic_and_author, topic_and_journal, topic_and_article]
evaluation_names = ['no_topic_run', 'topicLOW_run', 'topicMID_run', 'topicHIGH_run', 'no_topic_alphaMID_run', 'topicLOW_alphaMID_run', 'topicMID_alphaMID_run', 'no_topic_alphaHIGH_run', 'topicLOW_alphaHIGH_run', 'topic_and_author', 'topic_and_journal', 'topic_and_article']


for ev in evaluation:
    break
    print("running evaluation: ", ev)

    result = utils.article_authority_score(citation_weight_adj_matrix, paper_author_adj_matrix, paper_journal_adj_matrix, cuis_adj_matrix, time_difference, article_time_difference, article_years, cites, cited, author_papers, topic_papers, journal_papers, paper_authors, paper_journals, paper_topics,
    alpha=ev.alpha, beta=ev.beta, gamma = ev.gamma, delta = ev.delta, omega=ev.omega, sigma=ev.sigma)

    np.save('evaluation/'+evaluation_names[evaluation.index(ev)]+'.npy', result)

    print("results saved")



PR = metric(alpha=0.8, beta=0.0, gamma=0.0, delta=0.0, omega=0.0, sigma=0.1)
PR_author = metric(alpha=0.5, beta=0.3, gamma=0.0, delta=0.0, omega=0.0, sigma=0.1)
PR_journal = metric(alpha=0.5, beta=0.0, gamma=0.3, delta=0.0, omega=0.0, sigma=0.1)
PR_article = metric(alpha=0.5, beta=0.0, gamma=0.0, delta=0.0, omega=0.3, sigma=0.1)
PR_author_journal = metric(alpha=0.4, beta=0.2, gamma=0.2, delta=0.0, omega=0.0, sigma=0.1)
PR_author_article = metric(alpha=0.4, beta=0.2, gamma=0.0, delta=0.0, omega=0.2, sigma=0.1)
PR_journal_article = metric(alpha=0.4, beta=0.0, gamma=0.2, delta=0.0, omega=0.2, sigma=0.1)
PR_author_journal_article = metric(alpha=0.2, beta=0.2, gamma=0.2, delta=0.0, omega=0.2, sigma=0.1)


PR_topic = metric(alpha=0.5, beta=0.0, gamma=0.0, delta=0.3, omega=0.0, sigma=0.1)
PR_author_topic = metric(alpha=0.3, beta=0.2, gamma=0.0, delta=0.3, omega=0.0, sigma=0.1)
PR_journal_topic = metric(alpha=0.3, beta=0.0, gamma=0.2, delta=0.3, omega=0.0, sigma=0.1)
PR_article_topic = metric(alpha=0.3, beta=0.0, gamma=0.0, delta=0.3, omega=0.2, sigma=0.1)
PR_author_journal_topic = metric(alpha=0.2, beta=0.1, gamma=0.1, delta=0.3, omega=0.2, sigma=0.1)
PR_author_article_topic = metric(alpha=0.2, beta=0.1, gamma=0.0, delta=0.3, omega=0.2, sigma=0.1)
PR_journal_article_topic = metric(alpha=0.2, beta=0.0, gamma=0.1, delta=0.3, omega=0.2, sigma=0.1)
PR_author_journal_article_topic = metric(alpha=0.2, beta=0.133, gamma=0.133, delta=0.2, omega=0.133, sigma=0.1)


evaluation = [PR, PR_author, PR_journal, PR_article, PR_author_journal, PR_author_article, PR_journal_article, PR_author_journal_article, PR_topic, PR_author_topic, PR_journal_topic, PR_article_topic, PR_author_journal_topic, PR_author_article_topic, PR_journal_article_topic, PR_author_journal_article_topic]

evaluation_names = ['PR', 'PR_author', 'PR_journal', 'PR_article', 'PR_author_journal', 'PR_author_article', 'PR_journal_article', 'PR_author_journal_article', 'PR_topic', 'PR_author_topic', 'PR_journal_topic', 'PR_article_topic', 'PR_author_journal_topic', 'PR_author_article_topic', 'PR_journal_article_topic', 'PR_author_journal_article_topic']



Topic = metric(alpha=0.1, beta=0.0, gamma=0.0, delta=0.7, omega=0.0, sigma=0.1)
Author = metric(alpha=0.1, beta=0.7, gamma=0.0, delta=0.0, omega=0.0, sigma=0.1)
Journal = metric(alpha=0.1, beta=0.0, gamma=0.7, delta=0.0, omega=0.0, sigma=0.1)
Article = metric(alpha=0.1, beta=0.0, gamma=0.0, delta=0.0, omega=0.7, sigma=0.1)

new_evaluation = [Topic, Author, Journal, Article]

new_evaluation_names = ['Topic', 'Author', 'Journal', 'Article']

for ev in new_evaluation:
    print("running evaluation: ", ev)

    result = utils.article_authority_score(citation_weight_adj_matrix, paper_author_adj_matrix, paper_journal_adj_matrix, cuis_adj_matrix, time_difference, article_time_difference, article_years, cites, cited, author_papers, topic_papers, journal_papers, paper_authors, paper_journals, paper_topics,
    alpha=ev.alpha, beta=ev.beta, gamma = ev.gamma, delta = ev.delta, omega=ev.omega, sigma=ev.sigma)

    np.save('evaluation/'+new_evaluation_names[new_evaluation.index(ev)]+'.npy', result)

    print("results saved")
