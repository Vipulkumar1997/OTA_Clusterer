#!/bin/sh
mkdir ~/ota-clusterer
cd ~/ota-clusterer
virtualenv -p python3 ota-clusterer
source ota-clusterer/bin/activate
cd -
pip install -r requirements.txt

