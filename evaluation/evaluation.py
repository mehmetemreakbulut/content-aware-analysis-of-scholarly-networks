import numpy as np
import json
import spearman

papers = json.load(open('../dataset_papers.json', 'r'))
try:
    papers_index = json.load(open('useful/papers_index.json', 'r'))
except:
    papers_index = {}
    for i, paper in enumerate(papers):
        papers_index[paper['paperId']] = i

def get_max_papers(result_list,count=100):
    max_papers = []
    for i in range(count):
        max_index = int(np.argmax(result_list))
        max_papers.append(max_index)
        result_list[max_index] = -1
    return max_papers

from dataclasses import dataclass
@dataclass
class metric:
    alpha: float
    beta: float
    gamma: float
    delta: float
    omega: float
    sigma: float

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


runs = [PR, PR_topic, PR_author, PR_journal, PR_article, PR_author_topic, PR_author_journal, PR_author_article, PR_journal_topic, PR_journal_article,  PR_article_topic,  PR_author_journal_topic, PR_author_journal_article,  PR_author_article_topic, PR_journal_article_topic, PR_author_journal_article_topic]

evaluation_explanation = ['PR', 'PR_topic', 'PR_author', 'PR_journal', 'PR_article', 'PR_author_topic', 'PR_author_journal', 'PR_author_article', 'PR_journal_topic', 'PR_journal_article',  'PR_article_topic',  'PR_author_journal_topic', 'PR_author_journal_article',  'PR_author_article_topic', 'PR_journal_article_topic', 'PR_author_journal_article_topic']


top = {}
scores = {}
for i in range(len(evaluation_explanation)):
    file = ''+evaluation_explanation[i]+'.npy'

    run = np.load(file)

    #get top 100 papers
    top[i] = get_max_papers(run, 100)
    scores[i] = run

#compare all runs by comparing common papers
results = {}
for i in range(len(evaluation_explanation)):
    for j in range(i, len(evaluation_explanation)):
        common_papers = set(top[i]).intersection(set(top[j]))
        results[(i,j)] = len(common_papers)


#order results by number of common papers
results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

for key, value in results.items():
    print("{} vs {}: {}".format(evaluation_explanation[key[0]], evaluation_explanation[key[1]], value))



correlations = {}
import os
try:
    with open('correlations.json') as f:
        correlations = json.load(f)
    #make keys tuples
    correlations = {eval(key): value for key, value in correlations.items()}
except:
    print("Calculating correlations")
    for i in range(len(evaluation_explanation)):
        for j in range(i, len(evaluation_explanation)):
            correlation = spearman.pearson_correlation_rank(scores[i], scores[j])

            correlations[(i,j)] = correlation
            correlations[(j,i)] = correlation

    correlations = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1], reverse=True)}

    #save correlations
    #create file
    #make keys string
    correlations_str = {}
    for key, value in correlations.items():
        correlations_str[str(key)] = value

    with open('correlations.json', 'w') as f:
        json.dump(correlations_str, f)



for key, value in correlations.items():
    print("{} vs {}: {}".format(evaluation_explanation[key[0]], evaluation_explanation[key[1]], value))


#confusion matrix with matplotlib
import matplotlib.pyplot as plt

#confusion matrix
confusion_matrix = np.zeros((len(evaluation_explanation), len(evaluation_explanation)))

print(correlations)
for i in range(len(evaluation_explanation)):
    for j in range(i, len(evaluation_explanation)):
        #confusion_matrix[i][j] = results[(i,j)]
        confusion_matrix[j][i] = correlations[(i,j)]

from sklearn import metrics


plt.imshow(confusion_matrix, cmap='Blues')
#add names to the matrix
plt.xticks(range(len(evaluation_explanation)), evaluation_explanation, rotation=90)
plt.yticks(range(len(evaluation_explanation)), evaluation_explanation)
plt.colorbar()
#plt.show()



#compare PR+topic with other runs in a scatter plot
import matplotlib.pyplot as plt

name =  'PR'
print(evaluation_explanation)
index = evaluation_explanation.index(name)

related_correlations = {}
for i in range(len(evaluation_explanation)):
    related_correlations[i] = correlations[(index, i)]

y_axis_names = []
names = []
scores = []
for name in evaluation_explanation:
    if name == 'PR':
        continue
    index = evaluation_explanation.index(name)
    scores.append(related_correlations[index])
    names.append(name)

#make a bar char correlation vs names
plt.plot(names, scores)
plt.xticks(rotation=90)
#add annotations
for i, txt in enumerate(scores):
    plt.annotate(txt, (names[i], scores[i]))

plt.grid()
#plt.show()


Topic = metric(alpha=0.1, beta=0.0, gamma=0.0, delta=0.7, omega=0.0, sigma=0.1)
Author = metric(alpha=0.1, beta=0.7, gamma=0.0, delta=0.0, omega=0.0, sigma=0.1)
Journal = metric(alpha=0.1, beta=0.0, gamma=0.7, delta=0.0, omega=0.0, sigma=0.1)
Article = metric(alpha=0.1, beta=0.0, gamma=0.0, delta=0.0, omega=0.7, sigma=0.1)

new_evals = [Topic, Author, Journal, Article, PR]

new_eval_names = ['Topic', 'Author', 'Journal', 'Article' , 'PR']

top10 = {}
scores10 = {}
for i in range(len(new_evals)):
    file = ''+new_eval_names[i]+'.npy'

    run = np.load(file)

    #get top 100 papers
    top10[i] = get_max_papers(run, 100)


    scores10[i] = run

#save to same file the top 10 paper names
top1o_with_papers = {}
title_id = {}
for i in range(len(new_evals)):
    paper_n = []
    for paper in top10[i]:
        paper_n.append(papers[paper]['paperId'])
        title_id[papers[paper]['paperId']] = papers[paper]['title']
    top1o_with_papers[new_eval_names[i]] = paper_n



#compare all runs by comparing common papers
results10 = {}
for i in range(len(new_evals)):
    for j in range(i, len(new_evals)):
        common_papers = set(top10[i]).intersection(set(top10[j]))
        results10[(i,j)] = len(common_papers)


#order results by number of common papers
results10 = {k: v for k, v in sorted(results10.items(), key=lambda item: item[1], reverse=True)}

for key, value in results10.items():

    print("{} vs {}: {}".format(new_eval_names[key[0]], new_eval_names[key[1]], value))

import pandas as pd
citations = pd.read_csv('../dataset_citations.csv')

ids = []

for key, value in top1o_with_papers.items():
    ids.extend(value)

ids = set(ids)


all_ids_citation = {}
for paper in papers:
    all_ids_citation[paper['paperId']] =0

for row in citations.iterrows():
    all_ids_citation[row[1]['target']] += 1

#save citation count



# rank papers by citation count
sorted_citation = {k: v for k, v in sorted(all_ids_citation.items(), key=lambda item: item[1], reverse=True)}

#save to file
with open('citation_count_all.json', 'w') as f:
    json.dump(sorted_citation, f)

#rank papers by citation count
paper_ranks = {}

for i, key in enumerate(sorted_citation.keys()):
    paper_ranks[key] = i

citations = []
scores10_new = [[], [], [], [], []]
for i, p in enumerate(papers):
    if all_ids_citation[p['paperId']] < 5:
        continue
    citations.append(all_ids_citation[p['paperId']])
    for j in range(len(new_evals)):
        scores10_new[j].append(scores10[j][i])

#correlation between citation count and evaluations
print(len(citations))
correlations = []
for i in range(len(new_evals)):
    correlation = spearman.pearson_correlation_rank(scores10_new[i], citations)
    correlations.append(correlation)

for i in range(len(new_evals)):
    print("{} vs citation count: {}".format(new_eval_names[i], correlations[i]))
