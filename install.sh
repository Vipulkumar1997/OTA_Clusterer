#!/bin/sh
echo Bootstrapping stuff for OTA Clusterer...
echo ...creating folders, install dependencies and more...

if command -v python3 &>/dev/null; then
  echo Python is available...
else
  echo Python is not installed, abort installation! Pls install Python3..
  exit 1
fi

if command -v pip &>/dev/null; then
  echo PIP is available...
else
  echo PIP is not installed, abort installation! Pls install PIP...
  exit 1
fi

echo
echo create folder ota-clusterer for virtualenv at home folder...
echo

mkdir ~/ota-clusterer
cd ~/ota-clusterer

echo
echo install virtualenv via PIP
echo
pip install virtualenv
virtualenv -p python3 ota-clusterer
source ota-clusterer/bin/activate
cd -
echo
echo install requirements via PIP
echo
pip install -r requirements.txt
cd ~/ota-clusterer
source ota-clusterer/bin/activate

echo
echo download nltk data for doc2vec
echo

python -m nltk.downloader all-corpora

if [ "$(uname)" == "Darwin" ]; then
  echo installation takes place on a Mac, some changes to matplotlib necessary...
  DIRECTORY=$HOME/.matplotlib/
  echo $DIRECTORY
  if [ ! -d "$DIRECTORY" ] ; then
    echo creating matplotlib folder
    mkdir ~/.matplotlib
  else
    echo matplotlib folder already there...do nothing
  fi

  MATPLOTLIBRC=$HOME/.matplotlib/matplotlibrc
  if [ ! -f "$MATPLOTLIBRC" ]; then
    echo creating matplotlibrc file...
    echo backend: TkAgg > $HOME/.matplotlib/matplotlibrc

  else
    echo matplotlibrc file already exists...do nothing
  fi

fi
cd -
echo
echo successfully installed OTA Clusterer
