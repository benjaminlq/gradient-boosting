#!/usr/bin/env bash

cd data
COMPETITION="predict-student-performance-from-game-play"
ZIPFILE="${COMPETITION}.zip"

# kaggle competitions download -c $COMPETITION
unzip $ZIPFILE
#rm $ZIPFILE