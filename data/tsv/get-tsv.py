#! /bin/env python

from __future__ import print_function
import urllib, os
import gzip
from subprocess import call

URLS = [
"https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-Bk.tsv",
"https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-B2k.tsv",
"https://graphchallenge.s3.amazonaws.com/snap/soc-Epinions1/soc-Epinions1_adj.tsv",
"https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale22-ef16/graph500-scale22-ef16_adj.tsv.gz",
"https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale21-ef16/graph500-scale21-ef16_adj.tsv.gz",
"https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale20-ef16/graph500-scale20-ef16_adj.tsv.gz"
]

def get_remote_size(url):
  site = urllib.urlopen(url)
  meta = site.info()
  return int(meta.getheaders("Content-Length")[0])

def get_local_size(path):
  try:
    return os.stat(path).st_size
  except OSError:
    return 0

def download(url, dst):
  print(url, "->", dst)
  urllib.urlretrieve(url, dst)

for url in URLS:
  remoteSize = get_remote_size(url)
  localFile = os.path.basename(url)
  localSize = get_local_size(localFile)
  if localSize != remoteSize:
    print("getting", remoteSize/1024, "KB")
    download(url, localFile)
  else:
    print("already have", url)

  _, file_extension = os.path.splitext(localFile)
  if file_extension == ".tar.gz":
    call(['tar', '-xf', localFile])
  elif file_extension == ".gz":
    cmd = ['gunzip', "-k", "-f", localFile]
    print("running", " ".join(cmd))
    call(cmd)



