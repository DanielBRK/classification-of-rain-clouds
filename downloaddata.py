# python downloaddata.py

import os
os.system("pip install --upgrade --no-cache-dir gdown")
import gdown

url = 'https://drive.google.com/u/0/uc?id=1oG5lHjIkPktpN-XEW-1sJhmXPxFqL4j8&export=download'
out_path = 'data.zip'
id1="1oG5lHjIkPktpN-XEW-1sJhmXPxFqL4j8"

gdown.download(url,out_path)
os.system('unzip data_orig.zip')
