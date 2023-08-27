import pandas as pd
import requests
import os

df = pd.read_csv("cgheorgh-challenge-filenames.csv", names=["File","RIP"])
df = df.drop(columns=['RIP'])
for image in df["File"]:
 os.environ['no_proxy'] = ""
 url = 'https://cs7ns1.scss.tcd.ie?shortname=cgheorgh&myfilename='+image
 response = requests.get(url, proxies={"proxies": 'http://proxy.scss.tcd.ie:8080'})
 open(image,'wb').write(response.content)
