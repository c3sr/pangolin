#! /bin/bash

# To run from command line, ensure DOCKER_CUDA variable is set first

if [ $CI ]; then
set -x
fi

if [[ -z ${DRY_RUN+x} ]]; then
    DRY_RUN=1
fi

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}


# Determine git branch
if [ $CI ]; then
    source ~/.bashrc
    cd ${TRAVIS_BUILD_DIR}
    BRANCH=$TRAVIS_BRANCH
else
    BRANCH=`git rev-parse --abbrev-ref HEAD`
fi
# replace / with -
BRANCH="${BRANCH//\//-}"


# if DOCKER_ARCH is set
if [[ ! -z ${DOCKER_ARCH+x} ]]; then
    ARCH=${DOCKER_ARCH}
else
    ARCH=`uname -m`
    if [ $ARCH == x86_64 ]; then
        ARCH=amd64
    fi
fi

SHA=`git rev-parse --short HEAD`

REPO=c3sr/pangolin
TAG=`if [ "$BRANCH" == "master" ]; then echo "latest"; else echo "${BRANCH}"; fi`
TAG="$TAG-$SHA"

# untracked files
git ls-files --exclude-standard --others
DIRTY=$?

if [ "$DIRTY" == 0 ]; then
# staged changes, not yet committed
git diff-index --quiet --cached HEAD --
DIRTY=$?
fi

if [ "$DIRTY" == 0 ]; then
# working tree has changes that could be staged
git diff-files --quiet
DIRTY=$?
fi

if [ "$DIRTY" != 0 ]; then
    TAG=$TAG-dirty
fi


if [[ ! -z ${DOCKER_ARCH+x} ]]; then
    set +x
    echo "$DOCKER_PASSWORD" | or_die docker login --username "$DOCKER_USERNAME" --password-stdin
    set -x

    if [ "$ARCH" == amd64 ]; then # if amd64, build on travis
        or_die docker build -f $ARCH.cuda${DOCKER_CUDA}.Dockerfile -t $REPO:$ARCH-cuda${DOCKER_CUDA}-$TAG .
        or_die docker push $REPO
    elif [ "$ARCH" == ppc64le ]; then # if ppc64le, build on rai

rai_build="rai:
  version: 0.2
resources:
  cpu:
    architecture: ppc64le
  network: true
commands:
  build_image:
    image_name: $REPO:$ARCH-cuda${DOCKER_CUDA}-$TAG
    dockerfile: \"./$ARCH.cuda${DOCKER_CUDA}.Dockerfile\"
    no_cache: true
    push:
      push: true
"

        echo "$rai_build" > rai_build.yml
        or_die rai -d -v -p . -q rai_ppc64le_osu
    fi
else
    if [[ -z ${DOCKER_CUDA+x} ]]; then
        echo "please set DOCKER_CUDA"
        exit 1
    fi
    if [[ $DRY_RUN == 1 ]]; then
        echo "would run" docker build -f $ARCH.cuda${DOCKER_CUDA}.Dockerfile -t $REPO:$ARCH-cuda${DOCKER_CUDA}-$TAG .
        echo "would run" docker push ${REPO}:${ARCH}-cuda${DOCKER_CUDA}-$TAG
        echo run with DRY_RUN=0 to build and push
    else
        docker build -f $ARCH.cuda${DOCKER_CUDA}.Dockerfile -t $REPO:$ARCH-cuda${DOCKER_CUDA}-$TAG .
        docker push ${REPO}:${ARCH}-cuda${DOCKER_CUDA}-$TAG
    fi
fi

set +x
exit 0