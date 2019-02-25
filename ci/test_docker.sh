#! /bin/bash

nice -n20 docker build -f test_cuda80-ubuntu1404.Dockerfile . -t pangolin-test
nice -n20 docker build -f test_cuda80-ubuntu1604.Dockerfile . -t pangolin-test
nice -n20 docker build -f test_cuda92-ubuntu1804.Dockerfile . -t pangolin-test
nice -n20 docker build -f test_cuda100-ubuntu1804.Dockerfile . -t pangolin-test