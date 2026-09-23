#!/bin/bash
CORE=~/lbcf/Code/Model/LBCF/UDCF_RCT/core
OUT=~/lbcf/Code/Model/LBCF/output
IN=~/lbcf/Data/hillstrom/hillstrom_input.txt
cd "$CORE/build" || exit 1

run () {
  MNS=$1
  IP=$2
  TAG=$3
  echo "=============================================================="
  echo " min_node_size = $MNS   |   imbalance_penalty = $IP"
  echo "=============================================================="
  START=$(date +%s)
  ./UDCF_RCT $MNS $IP $IN $OUT/hill_$TAG 2>&1 | grep -E "DIAG_NODES|DIAG_SPLIT_TOTAL|DIAG_SPLIT_FREQ"
  END=$(date +%s)
  echo "tempo: $((END-START))s"
  F=$OUT/hill_$TAG
  echo "predicoes: $(wc -l < $F) linhas | $(sort -u $F | wc -l) unicas"
  echo ""
}

run 50 0.01 a_mns50_ip001
run 5  0.01 b_mns5_ip001
run 50 0    c_mns50_ip0
run 5  0    d_mns5_ip0
