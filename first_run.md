alpha = page_rank_scores
beta = authority_from_authors
gamma = authority_from_journals
delta = authority_from_topics
omega = authority_from_articles
sigma = time_vector[i]
(1 - alpha - beta - gamma - delta - omega - sigma) = (1/n_P)


first_run : time_p = 0.62, alpha = 0.3, beta = 0.2, gamma = 0.1, delta=0.1, omega = 0.2, sigma = 0.1, epsilon = 0.0001
second_run: time_p = 0.62, alpha=0.7, beta=0.1, gamma = 0.05, delta = 0, omega=0.05, sigma=0.1
third_run: time_p = 0.62,  alpha=0.85, beta=0.0, gamma = 0.00, delta = 0, omega=0.00, sigma=0.0 -> only pagerank + RandomJump
forth_run: time_p = 0.62,  alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.3, omega=0.1, sigma=0.1
fifth_run: time_p = 0.62,  alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.3, sigma=0.1
sixth_run: time_p = 0.62,  alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.1, sigma=0.3
seventh_run: time_p = 0.62,  alpha=0.3, beta=0.05, gamma = 0.05, delta = 0.5, omega=0.05, sigma=0.05
eighth_run: time_p = 0.62,  alpha=0.05, beta=0.1, gamma = 0.1, delta = 0.4, omega=0.25, sigma=0.1
nineth_run: time_p = 0.62,  alpha=0.3, beta=0.1, gamma = 0.1, delta = 0.2, omega=0.2, sigma=0.0 -> no time, random jump
tenth_run: time_p = 0.62,  alpha=0.5, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.1, sigma=0.0
eleventh_run: time_p = 0.62,  alpha=0.6, beta=0.05, gamma = 0.05, delta = 0.05, omega=0.1, sigma=0.0
twelveth_run: time_p = 0.62,  alpha=0.65, beta=0.05, gamma = 0.05, delta = 0.05, omega=0.05, sigma=0.0
thirteenth_run: time_p = 0.62,  alpha=0.7, beta=0.05, gamma = 0.00, delta = 0.05, omega=0.05, sigma=0.0
fourteenth_run: time_p = 0.62,  alpha=0.75, beta=0.0, gamma = 0.00, delta = 0.05, omega=0.05, sigma=0.0
fifteenth_run: time_p = 0.62,  alpha=0.8, beta=0.0, gamma = 0.00, delta = 0.05, omega=0.00, sigma=0.0
sixteenth_run: time_p = 0.62,  alpha=0.1, beta=0.3, gamma = 0.3, delta = 0.0, omega=0.3, sigma=0.0
seventeenth_run: time_p = 0.62,  alpha=0.0, beta=0.3, gamma = 0.3, delta = 0.1, omega=0.3, sigma=0.0




0.1 -> random

no_topic_run : alpha = 0.1, beta = 0.2, gamma = 0.2, delta=0.0, omega = 0.2, sigma = 0.1 - low pagerank with no topic
topicLOW_run: alpha=0.1, beta=0.2, gamma = 0.2, delta = 0.1, omega=0.2, sigma=0.1 - low pagerank low topic
topicMID_run: alpha=0.1, beta=0.1, gamma = 0.1, delta = 0.4, omega=0.1, sigma=0.1 - low pagerank mid topic
topicHIGH_run: alpha=0.1, beta=0.0, gamma = 0.0, delta = 0.7, omega=0.0, sigma=0.1 - high topic


no_topic_alphaMID_run : alpha = 0.4, beta = 0.1, gamma = 0.1, delta=0.0, omega = 0.2, sigma = 0.1 - mid pagerank with no topic
topicLOW_alphaMID_run: alpha=0.4, beta=0.1, gamma = 0.1, delta = 0.1, omega=0.1, sigma=0.1 - mid pagerank with low topic
topicMID_alphaMID_run: alpha=0.4, beta=0.0, gamma = 0.0, delta = 0.4, omega=0.0, sigma=0.1 - mid pagerank with mid topic

no_topic_alphaHIGH_run : alpha = 0.7, beta = 0.03, gamma = 0.03, delta=0.0, omega = 0.03, sigma = 0.1 - high pagerank with no topic
topicLOW_alphaHIGH_run: alpha=0.7, beta=0.00, gamma = 0.00, delta = 0.1, omega=0.0, sigma=0.1 - high pagerank with low topic


topic_and_author : alpha = 0.1, beta = 0.3, gamma = 0.0, delta=0.4, omega = 0.0, sigma = 0.1 -
topic_and_journal: alpha = 0.1, beta = 0.0, gamma = 0.3, delta=0.4, omega = 0.0, sigma = 0.1 -
topic_and_article   : alpha = 0.1, beta = 0.0, gamma = 0.0, delta=0.4, omega = 0.3, sigma = 0.1




######
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


Topic = metric(alpha=0.1, beta=0.0, gamma=0.0, delta=0.7, omega=0.0, sigma=0.1)
Author = metric(alpha=0.1, beta=0.7, gamma=0.0, delta=0.0, omega=0.0, sigma=0.1)
Journal = metric(alpha=0.1, beta=0.0, gamma=0.7, delta=0.0, omega=0.0, sigma=0.1)
Article = metric(alpha=0.1, beta=0.0, gamma=0.0, delta=0.7, omega=0.0, sigma=0.1)
