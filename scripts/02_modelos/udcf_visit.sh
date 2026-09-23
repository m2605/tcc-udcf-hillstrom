#!/bin/bash
RES=/mnt/c/Users/m2292/Documents/tese-mba/resultados
CORE=~/lbcf/Code/Model/LBCF/UDCF_RCT/core
IN=~/lbcf/Data/hillstrom/hillstrom_input_visit.txt
OUT=~/lbcf/Code/Model/LBCF/output

cp "$RES/hillstrom_input_visit.txt" $IN
echo "entrada: $(wc -l < $IN) linhas"
cd "$CORE/build" || exit 1
echo ""

roda () {
  IP=$1; ST=$2; TAG=$3; ROT=$4
  printf "%-28s : " "$ROT"
  ./UDCF_RCT 50 $IP $ST $IN $OUT/visit_$TAG 2>&1 | grep -oE "internal_nodes=[0-9]+|stump_trees=[0-9]+" | tr '\n' ' '
  echo ""
}

roda 0.01 1 udcf_default   "UDCF default (ip=0.01)"
roda 0    1 udcf_ip0       "UDCF ip=0"
roda 0.01 0 abla_default   "Ablacao ip=0.01"
roda 0    0 abla_ip0       "Ablacao ip=0"

echo ""
echo "copiando para resultados/"
cp $OUT/visit_udcf_default $OUT/visit_udcf_ip0 $OUT/visit_abla_default $OUT/visit_abla_ip0 "$RES/"
ls -la "$RES"/visit_* 2>/dev/null | head
