#!/bin/bash

#set -x

while(true); do
    NOW=$(date)
    echo ===== $NOW
    #tar -zcvf B5_Clasificacion.tgz B5_Clasificacion/
    #md5sum B5_Clasificacion.tgz
    #git config --global user.email "angelgarcan@blinder.com"
    #git config --global user.name "AngelBlinder"
    #git add B5_Clasificacion.tgz
    git add .
    git commit -a -m "$NOW"
    git push https://angelgarcan:hoyjin3456@github.com/angelgarcan/public.git
    echo "Sleeping..."
    echo ""
    sleep 60
done