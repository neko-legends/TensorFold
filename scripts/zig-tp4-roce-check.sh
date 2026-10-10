#!/usr/bin/env bash
# The Zig engine's N-way RoCE transport on the four Sparks, in the GPU window (no model, no weights): run on the head.
#
#   CONFIG=config/tp4-zig.env WORKERS="<user>@<rank1-link-ip> <user>@<rank2-link-ip> <user>@<rank3-link-ip>" \
#       scripts/zig-tp4-roce-check.sh [OUTDIR]
#
# The live engines must be stopped first (the GPU window, docs/ZIG-TP4-PLAN.md's runbook): this script refuses to start
# while any dsv41-tf4-* or <ZIG_NAME>-r* container runs on a node. It runs only containers named zigtp4-check-r<rank>
# (--rm), from IMAGE (the zig-tp4 build: tf-tp-test is in it since this commit), one a node, rank r on WORKERS[r-1].
#
# Steps (STEPS="preflight roce-host suite bench" by default; add failfast for the fail-fast gate, ~16 runs):
#   preflight  each node's RDMA devices of TF_TP_ROCE_HCA: port state, the RoCE v2 GID the engine picks, its IPv4
#              and netdev prefix, memlock. PASS when every rank has the same subnets, one a function (the pairing).
#   roce-host  the RoCE wire and proxy at world 4 without a GPU: 1,500 gathers of 1 B to 1 MiB bit for bit (variable
#              lengths too), then host round trips by size.
#   suite      tf-tp-test suite over RoCE at world 4 (TF_TP_ROCE_FALLBACK=error: no silent NCCL): every collective
#              bit-exact against the host reference (gathers in rank order, sums in rank order), graph replays, the
#              load-time probe; the same on NCCL alongside (NCCL's float sums within rounding: ring order).
#   bench      NCCL vs RoCE: the engine's all-gather sizes (16 B, 1..128 rows of 10,240 B), eager and graphed, and one
#              86-exchange decode window; a table at the end.
#   failfast   each rank as the victim (fatal, kill) on RoCE: every rank must exit 70 (the victim 137 when killed).
# Output: OUTDIR (default results/zig-tp4-roce-<date>), one log a step and rank, and SUMMARY.txt.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config=${CONFIG:-config/tp4-zig.env}
[[ -f "$config" ]] || { echo "missing $config (copy config/tp4-zig.env.example)" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$config"
set +a
read -r -a WORKERS <<< "${WORKERS:?set WORKERS: the ssh targets of ranks 1, 2, 3 as in config/tp4.env}"
WORLD=${TP_WORLD:-4}
(( ${#WORKERS[@]} == WORLD - 1 )) || { echo "WORKERS has ${#WORKERS[@]} hosts, TP_WORLD=$WORLD needs $((WORLD - 1))" >&2; exit 2; }
for key in IMAGE TF_TP_MASTER NCCL_SOCKET_IFNAME; do
    [[ -n "${!key:-}" && "${!key}" != *'<'* ]] || { echo "set $key in $config" >&2; exit 2; }
done
HCAS=${TF_TP_ROCE_HCA:-}
OUT=${1:-results/zig-tp4-roce-$(date +%Y%m%d-%H%M%S)}
STEPS=${STEPS:-preflight roce-host suite bench}
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"
say() { echo "$*" | tee -a "$SUMMARY"; }
on() { # on RANK CMD: run CMD (a string) on that rank's node
    local r=$1; shift
    if [[ "$r" == 0 ]]; then bash -c "$*"; else ssh -n -o BatchMode=yes -o ConnectTimeout=10 "${WORKERS[$((r - 1))]}" "$*"; fi
}
on_in() { # on_in RANK CMD < input: `on` with this script's stdin passed through
    local r=$1; shift
    if [[ "$r" == 0 ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "${WORKERS[$((r - 1))]}" "$*"; fi
}
fails=0
verdict() { # NAME OK DETAIL
    if [[ "$2" == 1 ]]; then say "PASS $1: $3"; else say "FAIL $1: $3"; fails=$((fails + 1)); fi
}

# --- safety: nothing of the live service (or a Zig server) may run; the GPUs and the CX7 ports are ours alone -------
pattern="^(dsv41-tf4-|tensorfold-zig|${ZIG_NAME:-zigtp4}-)"
listing="docker ps --format '{{.Names}}' | grep -E $(printf '%q' "$pattern") | grep -vE '^zigtp4-check-r[0-9]' || true"
for ((r = 0; r < WORLD; r++)); do
    if ! running=$(on "$r" "$listing"); then
        echo "rank $r: ssh / docker failed; nothing was started" >&2
        exit 2
    fi
    if [[ -n "$running" ]]; then
        echo "rank $r still runs: $running. Stop the live engines first (the window runbook); nothing was started." >&2
        exit 2
    fi
done

PORT=29800
# step NAME BACKEND CMD...: tf-tp-test CMD on every rank at once (rank 0 last, it listens), logs OUT/NAME.rR.log
step() {
    local name=$1 backend=$2 r; shift 2
    PORT=$((PORT + 10))
    local envs=(-e TF_TP_WORLD="$WORLD" -e TF_TP_MASTER="$TF_TP_MASTER" -e TF_TP_PORT="$PORT"
                -e TF_COMM_BACKEND="$backend" -e TF_TP_ROCE_FALLBACK=error -e TF_DSV41_PLAN_TIMEOUT=300
                -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}")
    [[ -n "$HCAS" ]] && envs+=(-e TF_TP_ROCE_HCA="$HCAS")
    local k
    for k in NCCL_IB_HCA NCCL_IB_TC TF_TP_ROCE_TC TF_TP_ROCE_CPU TF_TP_ROCE_HCAS GLM53_TF_ROCE_MAX_KB TF_TP_MAILBOX_MAX_KB \
             GLM53_TF_ROCE_FAST TF_TP_ROCE_FAST GLM53_TF_ROCE_POLLERS GLM53_TF_ROCE_STAGGER_NS TF_TP_ROCE_BLOCKS GLM53_TF_ROCE_BLOCKS; do
        [[ -n "${!k:-}" ]] && envs+=(-e "$k=${!k}")
    done
    local pids=()
    for ((r = WORLD - 1; r >= 0; r--)); do
        local cmd
        cmd=$(printf '%q ' docker run --rm --init --name "zigtp4-check-r$r" --gpus all --ipc=host --network host \
            --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK "${envs[@]}" -e TF_TP_RANK="$r" \
            --entrypoint timeout "$IMAGE" 900 tf-tp-test "$@")
        ( on "$r" "$cmd"; echo "EXIT rank $r rc $? at $(date +%s.%N)" ) > "$OUT/$name.r$r.log" 2>&1 &
        pids+=($!)
        sleep 1
    done
    wait "${pids[@]}"
}
rc_of() { grep -oE '^EXIT rank [0-9]+ rc [0-9]+' "$1" | awk '{print $5}'; }

# --- preflight: the HCAs the engine will open, as roce_verbs.zig detects them (lowest IPv4-mapped RoCE v2 GID) ------
if [[ " $STEPS " == *" preflight "* ]]; then
    probe=$(cat <<'EOF'
set -u
devs="${1:-}"; [ -n "$devs" ] || devs=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',')
echo "memlock $(ulimit -l)"
for d in ${devs//,/ }; do
    d=${d%%:*}; p=/sys/class/infiniband/$d/ports/1
    [ -d "$p" ] || { echo "HCA $d missing"; continue; }
    state=$(cut -d' ' -f2 "$p/state" 2>/dev/null); layer=$(cat "$p/link_layer" 2>/dev/null)
    found=0
    for g in $(ls "$p/gids" | sort -n); do
        gid=$(cat "$p/gids/$g" 2>/dev/null) || continue
        case "$gid" in 0000:0000:0000:0000:0000:ffff:*) ;; *) continue;; esac
        type=$(cat "$p/gid_attrs/types/$g" 2>/dev/null)
        [ "$type" = "RoCE v2" ] || continue
        h=${gid#0000:0000:0000:0000:0000:ffff:}; a=$((16#${h%%:*})); b=$((16#${h##*:}))
        ip="$((a >> 8)).$((a & 255)).$((b >> 8)).$((b & 255))"
        nd=$(cat "$p/gid_attrs/ndevs/$g" 2>/dev/null)
        pre=$(ip -o -4 addr show dev "$nd" 2>/dev/null | awk -v ip="$ip" '$4 ~ "^"ip"/" {split($4, s, "/"); print s[2]}')
        echo "HCA $d state $state link $layer gid $g ip $ip netdev $nd prefix ${pre:-?}"
        found=1; break
    done
    [ $found = 1 ] || echo "HCA $d state $state link $layer gid none"
done
EOF
)
    say "== preflight (TF_TP_ROCE_HCA=${HCAS:-<auto>})"
    for ((r = 0; r < WORLD; r++)); do
        on_in "$r" "bash -s -- $(printf '%q' "$HCAS")" <<< "$probe" > "$OUT/preflight.r$r.txt" 2>&1 || true
        sed "s/^/  rank $r: /" "$OUT/preflight.r$r.txt" | tee -a "$SUMMARY"
    done
    # every rank: each HCA ACTIVE on Ethernet with an IPv4 RoCE v2 GID; the /24s (roce_verbs.zig pairs by the GID's
    # first three octets; roce.py by the netdev's prefix) one a function and the same set on every rank
    ok=1; detail=""; ref=""
    for ((r = 0; r < WORLD; r++)); do
        f="$OUT/preflight.r$r.txt"
        n=$(grep -c '^HCA ' "$f"); good=$(grep -cE '^HCA .* state ACTIVE link Ethernet gid [0-9]+ ip ' "$f")
        nets=$(grep -oE ' ip [0-9]+\.[0-9]+\.[0-9]+' "$f" | awk '{print $2}' | sort)
        uniq_nets=$(sort -u <<< "$nets" | wc -l)
        [[ "$n" -ge 1 && "$good" == "$n" ]] || { ok=0; detail+="rank $r: $good of $n HCAs usable; "; }
        [[ "$uniq_nets" == "$(wc -l <<< "$nets")" ]] || { ok=0; detail+="rank $r: two HCAs on one /24 (the pairing keeps the listed order: check both nodes list them alike); "; }
        if [[ -z "$ref" ]]; then ref=$nets; elif [[ "$nets" != "$ref" ]]; then ok=0; detail+="rank $r's subnets differ from rank 0's; "; fi
        grep -qE '^memlock unlimited' "$f" || detail+="rank $r: host memlock is not unlimited (the container sets it: --ulimit memlock=-1); "
        grep -E 'prefix [0-9]+' "$f" | grep -vqE 'prefix 24$' && { ok=0; detail+="rank $r: a netdev prefix is not /24 (Zig pairs by /24, Python by the prefix: the pairing can diverge, so this step fails rather than warns); "; }
    done
    verdict preflight "$ok" "${detail:-every rank: the same subnets $(tr '\n' ' ' <<< "$ref")one a function}"
fi

# --- roce-host: wire + proxy at world 4, no GPU ----------------------------------------------------------------------
if [[ " $STEPS " == *" roce-host "* ]]; then
    say "== roce-host (wire + proxy, world $WORLD, no GPU)"
    step roce-host roce roce-host
    ok=1
    for ((r = 0; r < WORLD; r++)); do
        f="$OUT/roce-host.r$r.log"
        line=$(grep -h '^PASS roce-host\|^FAIL roce-host' "$f" | head -1)
        [[ "$(rc_of "$f")" == 0 && "$line" == PASS* ]] || { ok=0; line="${line:-no PASS line; tail: $(tail -3 "$f" | tr '\n' ' ')}"; }
        say "  rank $r: $line"
    done
    grep -h '^RESULT' "$OUT/roce-host.r0.log" | sed 's/^/  /' | tee -a "$SUMMARY"
    verdict roce-host "$ok" "1,500 gathers on every rank"
fi

# --- suite: every collective bit-exact over RoCE at world 4 (and NCCL alongside) -------------------------------------
if [[ " $STEPS " == *" suite "* ]]; then
    say "== suite (GPU, TF_COMM_BACKEND=roce, TF_TP_ROCE_FALLBACK=error)"
    step suite roce suite
    ok=1
    for ((r = 0; r < WORLD; r++)); do
        f="$OUT/suite.r$r.log"
        s=$(grep -h '^SUMMARY' "$f" | head -1)
        [[ "$(rc_of "$f")" == 0 && "$s" == *", 0 failed" ]] || ok=0
        say "  rank $r: ${s:-no SUMMARY (rc $(rc_of "$f")); tail: $(tail -3 "$f" | tr '\n' ' ')}"
        grep -h '^INFO rank\|HCA pairs\|tp roce probe\|RoCE unavailable' "$f" | sed 's/^/      /' | tee -a "$SUMMARY"
        grep -h '^FAIL' "$f" | head -5 | sed 's/^/      /' | tee -a "$SUMMARY"
    done
    grep -q 'backend roce' "$OUT/suite.r0.log" || { ok=0; say "  rank 0 did not come up on RoCE (see the [tensorfold] tp lines in suite.r0.log)"; }
    verdict suite "$ok" "every collective on RoCE and NCCL at world $WORLD (the per-rank SUMMARY lines above)"
fi

# --- bench: NCCL vs RoCE at the engine's sizes ----------------------------------------------------------------------
if [[ " $STEPS " == *" bench "* ]]; then
    say "== bench (NCCL vs RoCE, graphed microseconds a call, rank 0)"
    step bench roce bench
    ok=1
    for ((r = 0; r < WORLD; r++)); do [[ "$(rc_of "$OUT/bench.r$r.log")" == 0 ]] || ok=0; done
    grep -h '^RESULT' "$OUT/bench.r0.log" > "$OUT/bench.jsonl"
    python3 - "$OUT/bench.jsonl" <<'EOF' | tee -a "$SUMMARY" || ok=0
import json, sys
rows = [json.loads(l[len("RESULT "):]) for l in open(sys.argv[1])]
cell = {(r["backend"], r["op"], r["bytes"]): r for r in rows}
print(f"  {'op':<18}{'bytes':>9}{'rows':>6}{'nccl us':>10}{'roce us':>10}{'speedup':>9}")
seen = set()
for r in rows:
    key = (r["op"], r["bytes"])
    if key in seen or r["op"] not in ("engine_all_gather", "decode_window"):
        continue
    seen.add(key)
    n, o = cell.get(("nccl",) + key), cell.get(("roce",) + key)
    if not n or not o:
        continue
    f = "graph_us" if r["op"] == "engine_all_gather" else "window_us"
    print(f"  {r['op']:<18}{r['bytes']:>9}{r.get('rows', r.get('exchanges', '')):>6}{n[f]:>10.2f}{o[f]:>10.2f}{n[f] / o[f]:>8.2f}x")
EOF
    verdict bench "$ok" "table above; RoCE at or under NCCL up to the slot (GLM53_TF_ROCE_MAX_KB) is the expectation, the 128-row line is NCCL on both"
fi

# --- failfast: any rank's failure takes every rank down ----------------------------------------------------------------
if [[ " $STEPS " == *" failfast "* ]]; then
    say "== failfast (RoCE)"
    ok=1
    for mode in fatal kill; do
        for ((victim = 0; victim < WORLD; victim++)); do
            name=ff-$mode-$victim
            step "$name" roce failfast "$victim" "$mode"
            rcs=""; bad=0
            for ((r = 0; r < WORLD; r++)); do
                rc=$(rc_of "$OUT/$name.r$r.log"); rcs+="$rc "
                want=70; [[ "$r" == "$victim" && "$mode" == kill ]] && want=137
                # docker run reports the container's own code; a SIGKILLed tf-tp-test under timeout --init shows 137
                [[ "$rc" == "$want" ]] || bad=1
            done
            [[ $bad == 0 ]] || ok=0
            say "  $name: rc $rcs$([[ $bad == 0 ]] && echo '(as expected)' || echo '(want 70 everywhere, 137 for a killed victim)')"
        done
    done
    verdict failfast "$ok" "every victim, both modes"
fi

say "== $( ((fails == 0)) && echo 'ALL PASS' || echo "$fails step(s) FAILED") ($OUT)"
exit $((fails == 0 ? 0 : 1))
