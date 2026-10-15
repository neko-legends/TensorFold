"""bounded-json.py URL MODEL — qbench's 6 JSON-Schema tasks with every unbounded leaf bounded (strings maxLength 16, integers 0..999, numbers -> integer
range, arrays maxItems 3, open objects maxProperties 3 with names of at most 16 chars), so a truncated (gibberish) model must still finish inside
max_tokens if the server enforces the grammar: every reply then parses and validates against qbench's ORIGINAL schema."""
import copy, json, os, sys
sys.path.insert(0, os.path.join(os.environ.get("SPARK_BENCH", "/home/jun/git/spark-bench"), "bench"))
import qbench
def bound(s):
    s = copy.deepcopy(s)
    def walk(n):
        t = n.get("type")
        if t == "string" and "enum" not in n: n["maxLength"] = 16
        if t == "integer": n.setdefault("minimum", 0); n.setdefault("maximum", 999)
        if t == "number": n["minimum"] = -999; n["maximum"] = 999
        if t == "array":
            n.setdefault("maxItems", 3); walk(n["items"])
        if t == "object":
            for v in (n.get("properties") or {}).values(): walk(v)
            if isinstance(n.get("additionalProperties"), dict):
                n["maxProperties"] = 3; n["propertyNames"] = {"type": "string", "maxLength": 16}; walk(n["additionalProperties"])
    walk(s); return s
eng = qbench.Engine(sys.argv[1], sys.argv[2], "x", False, 300)
ok = 0
for t in qbench.load("json_tasks.json")["tasks"]:
    rf = {"type": "json_schema", "json_schema": {"name": t["name"], "strict": True, "schema": bound(t["schema"])}}
    r = eng.chat([{"role": "user", "content": t["prompt"]}], max_tokens=768, response_format=rf)
    try:
        errs = qbench.validate(json.loads(r["content"]), t["schema"])
    except Exception as e:
        errs = [f"unparsable: {e}"]
    ok += not errs
    print(t["name"], "OK" if not errs else errs[:2], r["finish"], repr(r["content"][:90]))
print(f"bounded json: {ok}/6 parse + validate against qbench's original schemas")
