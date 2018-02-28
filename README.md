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
- ...and hopefully in the future alternative word embedding methods (fastText is actually not supporting 'tagged documents')

### dimensionality reduction
- use t-sne to reduce the high dimensional documents vector model to 2D

### clusterer
focus of the research project is to evaluate different kind of clustering algorithms:
- Affinity Propagation
- K-Means
- K-Medoid
- DBSCAN
- Agglomerative Clustering (ward, complete and average)

results are visible in experiments folder...


## Install

Clone this project, go into the project folder and execute the install.sh script (Linux and MacOSX supported)
The script installs all dependencies into a seperated python virtualenv which is stored in your HOME folder.

```sh
cd OTA_Clusterer
./install.sh
```

IMPORTANT: Actually, there is a path-size limitation issue on Windows 10. You can't clone this repository successfully on a Windows machine. 

For Windows Users is there a first experimental version of an install script as well available and tested under Windows 10. Before running the script, have a look at the necessary instruction notes in the doc/windows folder:

Please keep in mind: This project has been developed under Linux. You get the best experience with a UNIX based machine.

```sh
cd OTA_Clusterer
.\install_windows.ps1
```
You can access all functionalities in your personal project as well, if you install it as Python module (tested with Linux and MacOSX, Windows should work as well):

```sh
cd OTA_Clusterer
python setup.py install

```

## Pre-Trained Models

If you would like to use some pre-trained models (doc2vec and t-SNE), you can download them here:
https://www.dropbox.com/sh/9bixl9hdd3j4qi8/AADX0k-iVi1C0WFgGDGtRoBua?dl=0

## Usage of the CLI

Clone this project and execute from the project folder following command to get all supported CLI options:
```sh
cd OTA Clusterer
python ota_clusterer/cli.py -h
```


## Built With

* [Gensim](https://radimrehurek.com/gensim/) - Topic Modelling
* [NLKT](http://www.nltk.org/) - Natural Language Tooltkit
* [scikit-learn](http://scikit-learn.org/stable/) - Machine Learning Library in Python
* [Scrapy](https://scrapy.org/) - Extracting data from website

## Authors

* **Sandro Cilurzo** [sandroci](https://github.com/sandroci)

See also the list of [contributors](https://github.com/sandroci/OTA_clusterer/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

Got inpsiration from following sources:

* [RaRe Technologies](https://rare-technologies.com/blog/)
* [GENSIM](https://markroxor.github.io/gensim/tutorials/)
* [o'reilly](https://github.com/oreillymedia/t-SNE-tutorial)
* [stefanpernes](https://github.com/stefanpernes/)
* [currie32](https://www.kaggle.com/currie32)

## IMPORTANT: Readme still work in progress... 
