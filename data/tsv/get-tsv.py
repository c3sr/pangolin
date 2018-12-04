#! /usr/bin/env python

from __future__ import print_function
import urllib, os
import gzip
from subprocess import call, check_output

URLS = [
"https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-Bk.tsv",
"https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-B1k.tsv",
"https://graphchallenge.s3.amazonaws.com/synthetic/gc3/Theory-16-25-81-B2k.tsv",
"https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale20-ef16/graph500-scale20-ef16_adj.tsv.gz",
"https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale21-ef16/graph500-scale21-ef16_adj.tsv.gz",
"https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale22-ef16/graph500-scale22-ef16_adj.tsv.gz",
]

SNAP_URLS = [
  "https://graphchallenge.s3.amazonaws.com/snap/soc-Epinions1/soc-Epinions1_adj.tsv",
  "https://graphchallenge.s3.amazonaws.com/snap/amazon0302/amazon0302_adj.tsv",
  "https://graphchallenge.s3.amazonaws.com/snap/roadNet-CA/roadNet-CA_adj.tsv",
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
  urllib.urlretrieve(url, dst)

def get_extracted_size(path):
  _, fileExtension = os.path.splitext(localFile)
  if ".gz" == fileExtension:
    cmd = ["gunzip", "-l", path]
    output = check_output(cmd)
    lines = output.splitlines()
    sz = int(lines[1].split()[1])
    return sz

for url in URLS + SNAP_URLS:
  remoteSize = get_remote_size(url)
  localFile = os.path.basename(url)
  localSize = get_local_size(localFile)
  if localSize != remoteSize:
    print(url, "->", localFile, "(", remoteSize/1024, "KB )")
    download(url, localFile)
  else:
    print(localFile, "already exists and is the expected size (", localSize, ")")

  _, file_extension = os.path.splitext(localFile)
  if file_extension == ".gz":
    extractedName = localFile[:-1 * len(file_extension)]
    expectSz = get_extracted_size(localFile)
    actualSz = get_local_size(extractedName)
    if expectSz != actualSz:
      cmd = ['gunzip', "-k", "-f", localFile]
      print("running", " ".join(cmd))
      call(cmd)
    else:
      print(extractedName, "already exists and is the expected size (", expectSz, ")")



