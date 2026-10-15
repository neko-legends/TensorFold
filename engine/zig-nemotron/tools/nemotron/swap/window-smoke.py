import json, time, urllib.request, base64, sys
U="http://forge:8100/v1/chat/completions"; M="NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
def chat(msgs, think, max_tokens=512, **kw):
    body={"model":M,"messages":msgs,"max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":think}, **kw}
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(U,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
    m=r["choices"][0]["message"]; return {"secs":round(time.time()-t,2),"usage":r.get("usage"),"finish":r["choices"][0].get("finish_reason"),
      "reasoning":(m.get("reasoning_content") or "")[:600],"content":(m.get("content") or "")[:1200]}
out=[]
for p,th in [("Explain in three sentences why the sky is blue.",False),
             ("Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",False),
             ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",True)]:
    r=chat([{"role":"user","content":p}],th,1024); r["prompt"]=p; r["thinking"]=th; out.append(r); print(json.dumps(r)[:1500],flush=True)
img=base64.b64encode(open("/home/jun/nest/tmp/spark-vision/bench/a4_table_chart.png","rb").read()).decode()
r=chat([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+img}},{"type":"text","text":"Describe this image and transcribe any table in it."}]}],False,1024)
r["prompt"]="image a4_table_chart.png"; out.append(r); print(json.dumps(r)[:2000],flush=True)
# 20K-token prompt: a long numbered list with a fact buried in it
lines=[f"Line {i}: the warehouse inventory record number {i} lists {i*7%97} crates of item code Z{i%113}." for i in range(1,900)]
lines.insert(450,"Line X: the secret code word for this document is PELICAN-42.")
r=chat([{"role":"user","content":"\n".join(lines)+"\n\nWhat is the secret code word for this document? Answer with just the code word."}],False,64)
r["prompt"]="20K-token needle"; out.append(r); print(json.dumps(r)[:1500],flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=1)
