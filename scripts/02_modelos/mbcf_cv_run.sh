#!/bin/bash
SC=/mnt/c/Users/m2292/AppData/Local/Temp/claude/C--Users-m2292-Documents-tese-mba/3b21be59-26c0-42ea-a9fc-3082fb63ab50/scratchpad
CORE=~/mbcf/MBCF_RCT/core
CV=~/mbcf/cv

rm -rf $CV && mkdir -p $CV
cp "$SC/mbcf_cv"/*.txt $CV/
echo "arquivos copiados: $(ls $CV | wc -l)"

cp "$SC/main_mbcf_cv.cpp" "$CORE/main.cpp"
cd "$CORE/build" || exit 1
make -j4 2>&1 | grep -iE "error" && exit 1
echo "compilado"
echo ""

for ARM in mens womens
do
  for F in 1 2 3 4 5
  do
    printf "%-8s dobra %s : " "$ARM" "$F"
    ./MBCF_RCT 0.01 $CV/train_${ARM}_${F}.txt $CV/test_${F}.txt $CV/pred_${ARM}_${F} \
      2>&1 | grep -oE "splits=[0-9]+" | tr '\n' ' '
    echo "linhas=$(wc -l < $CV/pred_${ARM}_${F})"
  done
done
echo ""
echo "concluido"
