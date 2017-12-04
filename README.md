# vm1-project-code
In this repository you will find the source code in relation to the ongoing research project called 'OTA Clusterer' at the Lucerne University of Applied Sciences of Information Technology. 

The primary goal is to examine methods and technologies to automically cluster online travel agencies (OTA). This should happen purely based on the public available website data and with no assistence of external domain expert knowledge. Actually the prototype consists of following software components which are purely written in Python (under construction): 

## components
### webcrawling
- custom scrapy module to crawl a given list of websites
- parsing all the HTML elements with BeautifulSoup and store each page of a given domain, as text file (documents)

### word embeddings
- preprocess the previously stored documents (tokenization, stemming, language detection etc.)
- create a document corpus based on the preprocessed data for english and german documents
- train a doc2vec model

### dimensionality reduction
- use t-sne to reduce the high dimensional documents vector model to 2D

### clusterer
focus of the research project is to evaluate different kind of clustering algorithms:
- Affinity Propagation
- K-Medoid
- DBSCAN



