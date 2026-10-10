#!/usr/bin/env bash
# Verify the logprobs build on the running four-Spark server (run AFTER a rollout; exit 0 = every check passed).
#
#   bash scripts/verify-logprobs.sh [BASE_URL]        # default http://forge:8000/v1
#
# Three checks, the same ones the rollout doc names:
#   A  the spec's logprobs request returns OpenAI-shaped probabilities (and a sane distribution);
#   B  nest/scripts/lib/typed_judge.py's forge probe produces a judgment (it reads token logprobs; skipped when
#      that library is not on this host);
#   C  a greedy chat reply is byte-for-byte the reply the tp4rot build gave (token ids, sha256 below) -- the
#      request that does not ask for probabilities must be untouched.
#
# It never restarts or stops anything: reads and one small chat request per check.
set -u
BASE="${1:-http://forge:8000/v1}"
CHAT="$BASE/chat/completions"
MODEL="DeepSeek-V4.1-Flash-TF"
GREEDY_SHA="03467e9a2d1ac0e5"          # tp4rot (2026-10-08, before this build): greedy token_ids' sha256
fail=0
say() { printf '%s %s\n' "$1" "$2"; }
note() { printf '   %s\n' "$1"; }

echo "== verify-logprobs: $CHAT =="

# -- A: logprobs -------------------------------------------------------------------------------------------------
LP=$(curl -s -m 180 "$CHAT" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"max_tokens\":4,\
\"logprobs\":true,\"top_logprobs\":20,\"chat_template_kwargs\":{\"enable_thinking\":false},\
\"messages\":[{\"role\":\"user\",\"content\":\"Answer true or false: 2+2=4\"}]}" | tr -d '\0')
if A=$(python3 - "$LP" <<'PY'
import json, math, sys
try:
    d = json.loads(sys.argv[1])
except Exception as exc:                                    # noqa: BLE001
    print(f"FAIL: the reply is not JSON ({exc}): {sys.argv[1][:200]}")
    raise SystemExit(1)
c = (d.get("choices") or [{}])[0]
lp = c.get("logprobs")
if not isinstance(lp, dict) or not lp.get("content"):
    print(f"FAIL: no choices[0].logprobs (keys {sorted(c)}) -- the server dropped the request's logprobs=True")
    raise SystemExit(1)
item = lp["content"][0]
for key in ("token", "logprob", "bytes", "top_logprobs"):
    if key not in item:
        print(f"FAIL: choices[0].logprobs.content[0] has no {key!r}")
        raise SystemExit(1)
tops = item["top_logprobs"]
if not tops or len(tops) > 20:
    print(f"FAIL: {len(tops)} top_logprobs (want 1..20)")
    raise SystemExit(1)
if item["logprob"] > 1e-6:
    print(f"FAIL: logprob {item['logprob']} > 0")
    raise SystemExit(1)
values = [t["logprob"] for t in tops]
if any(b > a + 1e-6 for a, b in zip(values, values[1:])):
    print(f"FAIL: top_logprobs not in descending order: {values}")
    raise SystemExit(1)
mass = sum(math.exp(v) for v in values)
if not 0.0 < mass <= 1.0 + 1e-6:
    print(f"FAIL: the top-20 mass is {mass} (want (0, 1])")
    raise SystemExit(1)
print(f"PASS: token {item['token']!r} logprob {item['logprob']:.4f}, {len(tops)} top_logprobs, "
      f"top {tops[0]['token']!r} {values[0]:.4f}, mass {mass:.4f}")
PY
); then say "[A]" "$A"; else say "[A]" "$A"; fail=1; fi

# -- B: the judge's own probe -----------------------------------------------------------------------------------
if [ -f /home/jun/nest/scripts/lib/typed_judge.py ]; then
    if B=$(timeout 300 python3 - <<'PY'
import json, sys
sys.path.insert(0, "/home/jun/nest/scripts/lib")
try:
    import typed_judge
    got = typed_judge.judge("2+2=4 is correct",
                            {"ok": {"type": "noul", "instructions": "Is the statement true?"}},
                            {"backend": "forge", "sensitive": True})
except Exception as exc:                                    # noqa: BLE001
    print(f"FAIL: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit(1)
text = json.dumps(got)[:200]
print(f"PASS: typed_judge returned {text}")
PY
    ); then say "[B]" "$B"; else say "[B]" "$B"; fail=1; fi
else
    say "[B]" "SKIP: /home/jun/nest/scripts/lib/typed_judge.py is not on this host"
fi

# -- C: a request that does not ask for probabilities -----------------------------------------------------------
G=$(curl -s -m 300 "$CHAT" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\
\"messages\":[{\"role\":\"user\",\"content\":\"List the first six prime numbers, comma separated, nothing else.\"}],\
\"max_tokens\":24,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},\"return_token_ids\":true}")
if C=$(python3 - "$G" "$GREEDY_SHA" <<'PY'
import hashlib, json, sys
d = json.loads(sys.argv[1])
choice = (d.get("choices") or [{}])[0]
ids = ((d.get("tensorfold") or {}).get("token_ids")) or []
if "logprobs" in choice:
    print("FAIL: a request without logprobs came back with a logprobs object")
    raise SystemExit(1)
got = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
content = (choice.get("message") or {}).get("content")
if got != sys.argv[2]:
    print(f"FAIL: greedy reply changed: token_ids {ids} ({got}), tp4rot's {sys.argv[2]}")
    raise SystemExit(1)
print(f"PASS: greedy reply unchanged ({got}): {content!r}")
PY
); then say "[C]" "$C"; else say "[C]" "$C"; fail=1; fi

echo "== verify-logprobs: $([ $fail -eq 0 ] && echo ALL PASS || echo FAILED) =="
exit $fail
