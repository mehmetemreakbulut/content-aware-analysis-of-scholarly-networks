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

runs = 17
top = {}
scores = {}
for i in range(1, runs+1):
    file_name = 'results/run{}.npy'.format(i)

    run = np.load(file_name)

    #get top 100 papers
    top[i] = get_max_papers(run, 100)

    #save scores
    scores[i] = run

#compare all runs by comparing common papers
results = {}
for i in range(1, runs+1):
    for j in range(i+1, runs+1):
        common_papers = set(top[i]).intersection(set(top[j]))
        results[(i,j)] = len(common_papers)

#order results by number of common papers
results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

for key, value in results.items():
    print("Run{} vs Run{}: {}".format(key[0], key[1], value))



# Calculate Spearman's rank correlation coefficient
# Scores are results of the runs

correlations = {}
for i in range(1, runs+1):
    for j in range(i+1, runs+1):
        correlation = spearman.pearson_correlation_rank(scores[i], scores[j])

        correlations[(i,j)] = correlation


correlations = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1], reverse=True)}

for key, value in correlations.items():
    print("Run{} vs Run{}: {}".format(key[0], key[1], value))



from dataclasses import dataclass

@dataclass
class metric:
    alpha: float
    beta: float
    gamma: float
    delta: float
    omega: float
    sigma: float

first_run = metric(alpha = 0.3, beta = 0.2, gamma = 0.1, delta=0.1, omega = 0.2, sigma = 0.1)
second_run= metric(alpha=0.7, beta=0.1, gamma = 0.05, delta = 0, omega=0.05, sigma=0.1)
third_run= metric(alpha=0.85, beta=0.0, gamma = 0.00, delta = 0, omega=0.00, sigma=0.0)
forth_run= metric(alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.3, omega=0.1, sigma=0.1)
fifth_run= metric(alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.3, sigma=0.1)
sixth_run= metric(alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.1, sigma=0.3)
seventh_run= metric(alpha=0.3, beta=0.05, gamma = 0.05, delta = 0.5, omega=0.05, sigma=0.05)
eighth_run= metric(alpha=0.05, beta=0.1, gamma = 0.1, delta = 0.4, omega=0.25, sigma=0.1)
nineth_run= metric(alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.2, omega=0.2, sigma=0.0)
tenth_run= metric(alpha=0.5, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.1, sigma=0.0)
eleventh_run= metric(alpha=0.6, beta=0.05, gamma = 0.05, delta = 0.05, omega=0.1, sigma=0.0)
twelveth_run= metric( alpha=0.65, beta=0.05, gamma = 0.05, delta = 0.05, omega=0.05, sigma=0.0)
thirteenth_run= metric(alpha=0.7, beta=0.05, gamma = 0.00, delta = 0.05, omega=0.05, sigma=0.0)
fourteenth_run= metric( alpha=0.75, beta=0.0, gamma = 0.00, delta = 0.05, omega=0.05, sigma=0.0)
fifteenth_run= metric( alpha=0.8, beta=0.0, gamma = 0.00, delta = 0.05, omega=0.00, sigma=0.0)
sixteenth_run= metric( alpha=0.1, beta=0.3, gamma = 0.3, delta = 0.0, omega=0.3, sigma=0.0)
seventeenth_run= metric(alpha=0.0, beta=0.3, gamma = 0.3, delta = 0.1, omega=0.3, sigma=0.0)

run_by_index = {1: first_run, 2: second_run, 3: third_run, 4: forth_run, 5: fifth_run, 6: sixth_run, 7: seventh_run, 8: eighth_run, 9: nineth_run, 10: tenth_run, 11: eleventh_run, 12: twelveth_run, 13: thirteenth_run, 14: fourteenth_run, 15: fifteenth_run , 16: sixteenth_run, 17: seventeenth_run}

#plot the correlations with respect to the metrics

#plot in 2d space by comparing correlation scores and difference of their alpha values

import matplotlib.pyplot as plt

correlation_scores = list(correlations.values())
alpha_diff = []
names = []
for key, value in correlations.items():
    index_1 = key[0]
    index_2 = key[1]
    alpha_diff.append(abs(run_by_index[index_1].alpha - run_by_index[index_2].alpha))
    #add scatter point labels for each point like (x,y)
    names.append("({}, {})".format(index_1, index_2))

plt.scatter(alpha_diff, correlation_scores)
for i, txt in enumerate(correlation_scores):
    plt.annotate(names[i], (alpha_diff[i], correlation_scores[i]))


plt.xlabel('Difference of Alpha Values')
plt.ylabel('Correlation Scores')
plt.title('Correlation Scores vs Difference of Alpha Values')
plt.show()

#plot in 2d space by comparing correlation scores and difference of their beta + gamma + omega values

sum_diff = []
names = []
for key, value in correlations.items():
    index_1 = key[0]
    index_2 = key[1]
    sum_diff.append(abs((run_by_index[index_1].beta + run_by_index[index_1].gamma + run_by_index[index_1].omega) - (run_by_index[index_2].beta + run_by_index[index_2].gamma + run_by_index[index_2].omega)))
    #add scatter point labels for each point like (x,y)
    names.append("({}, {})".format(index_1, index_2))

plt.scatter(sum_diff, correlation_scores)
for i, txt in enumerate(correlation_scores):
    plt.annotate(names[i], (sum_diff[i], correlation_scores[i]))

plt.xlabel('Difference of Beta + Gamma + Omega Values')
plt.ylabel('Correlation Scores')
plt.title('Correlation Scores vs Difference of Beta + Gamma + Omega Values')
plt.show()
