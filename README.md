
<br />
<div align="center">
  <a href="https://arxiv.org/abs/2411.00262">
    <img src="photos/icont.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Content-Aware Analysis of Scholarly Networks: A Case Study on CORD19 Dataset
</h3>

  <p align="center">
    Hybrid HITS Algorithm for Scholarly Network Analysis in the COVID-19 Domain
    <br />
    <a href="https://arxiv.org/abs/2411.00262"><strong>Explore the results »</strong></a>
    <br />
    <br />
    <a href="https://github.com/mehmetemreakbulut/content-aware-analysis-of-scholarly-networks/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/mehmetemreakbulut/content-aware-analysis-of-scholarly-networks/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

Please extract the zip files dataset_paper.json.zip and paper_with_annotation.json.zip.

Run the code by executing the `main.py` file. Creating matrix data from the dataset can take very long time, if you don't want to wait, download from the link below.

Locate the folder in the same directory as the `main.py` file.

For a real reproduction of the results, it is highly recommended to run the code from scratch.

Link: [Matrix](https://drive)


## Overview

This study investigates scholarly network analysis in the COVID-19 domain using the CORD-19 dataset to explore relationships among articles, researchers, and journals. The paper constructs a heterogeneous network structure, incorporating topic-based semantic information extracted via MedCAT, a medical concept annotation tool. Using a hybrid HITS algorithm, the approach assigns scores based on network structure and topic relationships, integrating semantic data with traditional citation-based metrics.

## Table of Contents

1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Methodology](#methodology)
    - [Heterogeneous Network](#heterogeneous-network)
    - [Link Weighting](#link-weighting)
    - [Topic Linking and Semantic Weighting](#topic-linking-and-semantic-weighting)
    - [Ranking Algorithm](#ranking-algorithm)
4. [Experiments and Results](#experiments-and-results)
5. [Appendices](#appendices)

---

## 1. Introduction

In scientific research, relationships between core elements such as articles, researchers, and institutions are crucial. Due to the rapid growth in publications, analyzing academic communities has become challenging, making it difficult for researchers to keep track of relevant literature. This paper addresses this challenge by proposing a hybrid method that incorporates topic-related information into citation analysis to produce more meaningful insights into the academic network.

## 2. Related Work

The paper builds on concepts from Social Network Analysis (SNA) and traditional citation metrics like the Impact Factor. Past approaches, including PageRank, CiteRank, and FutureRank, have focused on ranking scientific publications but often lack semantic context. This work integrates semantic meaning into the citation framework using Named Entity Recognition (NER) to account for topics within a citation network.

---

## 3. Methodology

### 3.1 Heterogeneous Network

The academic network is modeled as a heterogeneous graph with different types of nodes: articles, authors, journals, and topics. The heterogeneous network structure is defined as:

$$
G(V, E) = (V_{ar} \cup V_{au} \cup V_{ju} \cup V_{tp}, E_{ar-ar} \cup E_{ar-au} \cup E_{ar-ju} \cup E_{ar-tp} \cup E_{tp-tp})
$$

- **Node Types**: Articles (\(V_{ar}\)), authors (\(V_{au}\)), journals (\(V_{ju}\)), and topics (\(V_{tp}\)).
- **Edge Types**: Connections between articles, authors, journals, and topics. Topic nodes are connected to each other based on UMLS ontology.

### 3.2 Link Weighting

Different weighting schemes are applied to connections based on the type of relationship:
- **Article-Author**: Recognizes varying author contributions without over-relying on order of authorship or H-index alone.
- **Article-Journal**: Recognizes a journal’s influence, assuming equal quality among articles within prestigious journals.
- **Article-Article (Citation)**: Combines graph-based and semantic-based similarity measures for enhanced citation weight, using both network attributes and abstract-based semantic similarity.

#### Article-Article Weight Formula:

For two articles \(P1\) and \(P2\):
$$
S_n(P1, P2) = \alpha \frac{|(In_{P1} \cup Out_{P1}) \cap (In_{P2} \cup Out_{P2})|}{\sqrt{|In_{P1} \cup Out_{P1}| \times |In_{P2} \cup Out_{P2}|}} + \beta \frac{|A_{P1} \cap A_{P2}|}{\sqrt{|A_{P1}| \times |A_{P2}|}} + \gamma \cdot J_{P1-P2}
$$

where:
- \(In\) and \(Out\): Incoming and outgoing links,
- \(A\): Authors of the article,
- \(J\): Journal similarity indicator.

### 3.3 Topic Linking and Semantic Weighting

Semantic links between articles and topics are derived through Named Entity Recognition (NER) with MedCAT, based on UMLS ontology. The semantic similarity \(S_s(P1, P2)\) is computed as:

$$
S_s(P1, P2) = \frac{|TP1 \cap TP2|}{\sqrt{|TP1| \times |TP2|}}
$$

where \(TP1\) and \(TP2\) are sets of topics for each article.

The final weight \(S(P1, P2)\) between two articles is a combination of network and semantic similarity:

$$
S(P1, P2) = \alpha \cdot S_n + \beta \cdot S_s
$$

where \(\alpha\) and \(\beta\) are scaling parameters adjusted by median values.

### 3.4 Ranking Algorithm

The hybrid HITS algorithm is adapted for this network, assigning **hub** and **authority scores** to nodes.

#### Hub Score Calculation:

For authors, journals, topics, and articles, the hub score \(H\) is computed as:

$$
H(P_i) = \frac{\sum_{P_j \in N_i} wt(j) \cdot A(P_j)}{|N_i|}
$$

where \(N_i\) is the set of articles citing or cited by \(P_i\), \(A(P_j)\) is the authority score, and \(wt(j)\) is a time-aware weight.

#### Authority Score Calculation:

The authority score \(AS(P_i)\) is calculated by integrating scores from PageRank, author, journal, topic, and citation network as follows:

$$
AS(P_i) = \alpha \cdot PageRank(P_i) + \beta \cdot Author(P_i) + \gamma \cdot Journal(P_i) + \delta \cdot Topic(P_i) + \omega \cdot Article(P_i) + \sigma \cdot PTime_i
$$

where \(\alpha, \beta, \gamma, \delta, \omega, \sigma\) are constants representing the importance of each score source.

---

## 4. Experiments and Results

### 4.1 Dataset

The analysis utilizes the CORD-19 dataset, containing over 1 million COVID-19 research articles. Preprocessing removed duplicates, entries with missing information, and added citation data using Semantic Scholar API. The final dataset for analysis contained approximately 20,000 articles, over 120,000 authors, 2,900 journals, and over 200,000 citations.

### 4.2 Experimental Setup

Experiments were conducted by varying parameters \(\alpha\), \(\beta\), \(\gamma\), \(\delta\), \(\omega\), and \(\sigma\) to assess their impact on ranking performance. The parameter configurations maintained the sum constraint:

$$
\alpha + \beta + \gamma + \delta + \omega + \sigma = 1
$$

### 4.3 Key Findings

- **Correlation Analysis**: Adding topic information reduced correlation with PageRank-only results, suggesting enhanced differentiation.
- **Top Ranked Articles**: Semantic ranking prioritized review articles, showing broader topic coverage compared to citation-based rankings.
- **Citation Impact**: Lower correlation between topic-weighted settings and citation count indicated reduced citation bias.

---
