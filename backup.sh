#!/bin/bash
# https://mybinder.org/v2/gh/angelgarcan/mcdi/master
# https://mybinder.org/v2/gh/angelgarcan/public/master

#set -x

git config --global user.email "angelgarcan@blinder.com"
git config --global user.name "AngelBlinder"

while(true); do
    NOW=$(date +%FT%T)
    echo ===== $NOW
    #tar -zcvf B5_Clasificacion.tgz B5_Clasificacion/
    #md5sum B5_Clasificacion.tgz
    #git add B5_Clasificacion.tgz
    echo "*** DIFF:"
    git diff --stat
    echo "*** ADD:"
    git add "$1"
    echo "*** COMMIT:"
    git commit -a -m "$NOW: $1"
    echo "*** PUSH:"
    git push https://angelgarcan:${2}@github.com/angelgarcan/public.git
    echo "*** SLEEPING 300..."
    echo ""
    sleep 300
done