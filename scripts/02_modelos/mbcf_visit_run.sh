#!/bin/bash
SC=/mnt/c/Users/m2292/AppData/Local/Temp/claude/C--Users-m2292-Documents-tese-mba/3b21be59-26c0-42ea-a9fc-3082fb63ab50/scratchpad
CV=~/mbcf/cv_visit
rm -rf $CV && mkdir -p $CV
cp "$SC/mbcf_cv_visit"/*.txt $CV/
echo "arquivos: $(ls $CV | wc -l)"
cd ~/mbcf/MBCF_RCT/core/build || exit 1
for ARM in mens womens
do
  for F in 1 2 3 4 5
  do
    printf "%-8s dobra %s : " "$ARM" "$F"
    ./MBCF_RCT 0.01 $CV/train_${ARM}_${F}.txt $CV/test_${F}.txt $CV/pred_${ARM}_${F} 2>&1 | grep -oE "splits=[0-9]+"
  done
done
cp $CV/pred_* "$SC/mbcf_cv_visit/"
echo concluido
