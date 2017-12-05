# OTA Clusterer
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
- ...and alternative word embedding methods (TODO)

### dimensionality reduction
- use t-sne to reduce the high dimensional documents vector model to 2D
- PCA (TODO)

### clusterer
focus of the research project is to evaluate different kind of clustering algorithms:
- Affinity Propagation
- K-Means (TODO)
- K-Median (TODO)
- K-Medoid (TODO)
- EM (TODO)
- DBSCAN (TODO)
- DENCLUE (TODO)
- DIANI (TODO)
- ...and more to come

## Built With

* [Gensim](https://radimrehurek.com/gensim/) - Topic Modelling
* [NLKT] (http://www.nltk.org/) - Natural Language Tooltkit
* [scikit-learn](http://scikit-learn.org/stable/) - Machine Learning Library in Python
* [Scrapy](https://scrapy.org/) - Extracting data from website

## Authors

* **Sandro Cilurzo** [sandroci](https://github.com/sandroci)

See also the list of [contributors](https://github.com/sandroci/OTA_clusterer/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Got inpsiration from following sources:

* [RaRe Technologies](https://rare-technologies.com/blog/)
* [GENSIM](https://markroxor.github.io/gensim/tutorials/)
* [o'reilly] (https://github.com/oreillymedia/t-SNE-tutorial)
* [stefanpernes] (https://github.com/stefanpernes/)
* [currie32] (https://www.kaggle.com/currie32)

## IMPORTANT: - Still under construction 
