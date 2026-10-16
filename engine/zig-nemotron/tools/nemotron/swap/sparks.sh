# Sourced by the swap scripts: the four Sparks in rank order and rank 0's RoCE link address, kept out of the repo.
#   SPARKS="h0 h1 h2 h3" or ~/.config/nemotron-tf/hosts (one line, four ssh names; default spark1..spark4)
#   TF_TP_MASTER or ~/.config/nemotron-tf/tp-master (rank 0's address on the RoCE link)
read -r -a HOSTS <<< "${SPARKS:-$(cat "$HOME/.config/nemotron-tf/hosts" 2>/dev/null || echo spark1 spark2 spark3 spark4)}"
export R0=${HOSTS[0]}
MASTER=${TF_TP_MASTER:-$(cat "$HOME/.config/nemotron-tf/tp-master" 2>/dev/null)}
