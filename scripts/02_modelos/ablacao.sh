#!/bin/bash
SC=/mnt/c/Users/m2292/AppData/Local/Temp/claude/C--Users-m2292-Documents-tese-mba/f728b89b-8274-4692-8ac2-0528394a10c3/scratchpad
CORE=~/lbcf/Code/Model/LBCF/UDCF_RCT/core
OUT=~/lbcf/Code/Model/LBCF/output
IN=~/lbcf/Data/hillstrom/hillstrom_input.txt

cp "$SC/main_ablacao.cpp" "$CORE/main.cpp"
cd "$CORE/build" || exit 1
make -j4 2>&1 | grep -iE "error" && exit 1
echo "compilado"
echo ""

roda () {
  IP=$1
  ST=$2
  TAG=$3
  echo "=============================================================="
  ./UDCF_RCT 50 $IP $ST $IN $OUT/abl_$TAG 2>&1 | grep -E "REGRA|DIAG_NODES|DIAG_SPLIT_TOTAL|DIAG_SPLIT_FREQ"
  F=$OUT/abl_$TAG
  echo "predicoes: $(wc -l < $F) linhas | $(sort -u $F | wc -l) unicas"
  echo ""
}

echo "########## imbalance_penalty = 0.01 (default dos autores) ##########"
roda 0.01 1 udcf_ip001
roda 0.01 0 abla_ip001

echo "########## imbalance_penalty = 0 (criterio da Eq. 3) ##########"
roda 0 1 udcf_ip0
roda 0 0 abla_ip0
