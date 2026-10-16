import sys
rows=[]
for l in open(sys.argv[1]):
    if 'DTRACE' not in l: continue
    t=l.split(); a=int(t[3]); c=[float(x) for x in t[4:]]; rows.append((a,c))
segs={'code':rows[0:34],'prose':rows[34:114]+rows[114:218],'other':rows[218:331]}
B=[0,0.3,0.5,0.7,0.85,0.95,0.99,1.01]
for name,seg in segs.items():
    cnt=[0]*7;hit=[0]*7
    for a,c in seg:
        for j in range(min(len(c),8)):
            if a<j: break  # level j reached only if previous accepted
            b=next(i for i in range(7) if B[i]<=c[j]<B[i+1])
            cnt[b]+=1; hit[b]+= a>j
    print(name, ' '.join(f'[{B[i]:.2f},{B[i+1]:.2f}):{hit[i]}/{cnt[i]}={hit[i]/max(cnt[i],1):.2f}' for i in range(7)))
