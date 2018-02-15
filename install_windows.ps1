echo "Install dependencies for OTA Clusterer on Windows..."
cd ~/Documents/OTA_Clusterer
pip3.6 install -r requirements.txt
echo "download nltk data for doc2vec"
python -m nltk.downloader all-corpora
echo "successfully installed dependencies for OTA Clusterer on Windows"
Read-Host -Prompt "Press Enter to exit"