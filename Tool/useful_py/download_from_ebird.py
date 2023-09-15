import os
import pprint
import time
import urllib.error
import urllib.request

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            print(data)
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

if __name__ == '__main__':
    url = "https://media.ebird.org/catalog?taxonCode=minlor1"
    dst_path = ""
    download_file(url, dst_path)