#!/usr/bin/env python3
"""Seven serving gates against a running server (stdlib only; shapes.png next to this file).

    python3 verify.py [BASE_URL=http://localhost:8000/v1] [MODEL=DeepSeek-V4.1-Flash-1017-TF]

  chat_thinking_default  thinking on (the engine default): answer 'Paris' plus a reasoning field
  chat_thinking_off      chat_template_kwargs.enable_thinking=false: 'Paris' in a few tokens, no reasoning
  stream                 SSE stream of 'Count from 1 to 5' joins to '1, 2, 3, 4, 5'
  image                  shapes.png (a red square left of a blue circle) is read as such
  json_schema            response_format json_schema (strict) returns valid JSON with the three keys
  tools_thinking         a get_weather tool call for Tokyo, thinking on
  tools_nothink          the same tool call, thinking off

Temperature 0. Exit status 0 when all seven pass. On four Sparks with the 1017 pack (2026-10-17) the replies were:
'Paris' (+69 chars of reasoning), 'Paris' (2 tokens), '1, 2, 3, 4, 5', 'Red square, blue circle.',
{city, landmarks, population_millions}, get_weather {"city":"Tokyo","unit":"c"} twice.
"""
import base64, json, os, sys, urllib.request

B = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1").rstrip("/")
M = sys.argv[2] if len(sys.argv) > 2 else "DeepSeek-V4.1-Flash-1017-TF"
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def post(body, stream=False):
    body = {"model": M, "temperature": 0, **body}
    if stream:
        body["stream"] = True
    r = urllib.request.urlopen(urllib.request.Request(B + "/chat/completions", json.dumps(body).encode(),
                                                      {"content-type": "application/json"}), timeout=600)
    return list(r) if stream else json.load(r)


res = {}
ask = [{"role": "user", "content": "Capital of France? One word."}]
m = post({"messages": ask, "max_tokens": 300})["choices"][0]["message"]
reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
res["chat_thinking_default"] = ("Paris" in (m.get("content") or "") and len(reasoning) > 0, (m.get("content"), len(reasoning)))

d = post({"messages": ask, "max_tokens": 30, **OFF})
res["chat_thinking_off"] = ("Paris" in (d["choices"][0]["message"].get("content") or ""),
                            (d["choices"][0]["message"].get("content"), d["usage"]["completion_tokens"]))

txt, n = "", 0
for line in post({"messages": [{"role": "user", "content": "Count from 1 to 5, comma separated."}], "max_tokens": 40,
                  "stream_options": {"include_usage": True}, **OFF}, stream=True):
    line = line.decode().strip()
    if line.startswith("data: {"):
        j = json.loads(line[6:]); n += 1
        for c in j.get("choices", []):
            txt += c["delta"].get("content") or ""
res["stream"] = (txt.strip().rstrip(".") == "1, 2, 3, 4, 5", (txt, n))

png = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes.png"), "rb").read()
u = "data:image/png;base64," + base64.b64encode(png).decode()
d = post({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
          {"type": "text", "text": "Name the two shapes and their colours, left to right. Short."}]}], "max_tokens": 60, **OFF})
c = (d["choices"][0]["message"].get("content") or "").lower()
res["image"] = (all(w in c for w in ("red", "square", "blue", "circle")) and c.find("red") < c.find("blue"),
                (d["choices"][0]["message"].get("content"), d["usage"]["prompt_tokens"]))

sch = {"type": "object", "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"},
       "landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
       "required": ["city", "population_millions", "landmarks"], "additionalProperties": False}
d = post({"messages": [{"role": "user", "content": "Give JSON about Paris."}], "max_tokens": 200, **OFF,
          "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": sch, "strict": True}}})
c = d["choices"][0]["message"]["content"]
try:
    keys = sorted(json.loads(c)); res["json_schema"] = (keys == ["city", "landmarks", "population_millions"], keys)
except Exception:
    res["json_schema"] = (False, c[:200])

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                         "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
weather = [{"role": "user", "content": "What's the weather in Tokyo in celsius?"}]
for name, extra, mt in (("tools_thinking", {}, 400), ("tools_nothink", OFF, 200)):
    d = post({"messages": weather, "max_tokens": mt, "tools": tools, "tool_choice": "auto", **extra})
    calls = [(t["function"]["name"], t["function"]["arguments"]) for t in d["choices"][0]["message"].get("tool_calls") or []]
    ok = any(n == "get_weather" and "tokyo" in json.loads(a or "{}").get("city", "").lower() for n, a in calls)
    res[name] = (ok, (calls, d["choices"][0]["finish_reason"]))

passed = sum(ok for ok, _ in res.values())
for k, (ok, v) in res.items():
    print(f"{'PASS' if ok else 'FAIL'} {k}: {v}")
print(f"{passed}/{len(res)} passed")
sys.exit(passed != len(res))
