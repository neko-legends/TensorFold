import sys
V=[0,16.4121,19.2778,21.4814,23.685,25.8886,28.0922,29.7371,31.3819,33.0268,34.6716,36.3164,37.813,39.3096,40.8062,42.3028,41.7919]
L=0.741; OV=float(sys.argv[2]) if len(sys.argv)>2 else 1.5; SYNC=float(sys.argv[3]) if len(sys.argv)>3 else 0.1
rows=[]
for l in open(sys.argv[1]):
    if 'DTRACE' not in l: continue
    t=l.split(); a=int(t[3]); c=[float(x) for x in t[4:]]; c+= [0.0]*(9-len(c)); rows.append((a,c))
segs={'code':rows[0:34],'prose':rows[34:114],'story':rows[114:218],'explain':rows[218:293],'essay':rows[293:331]}
def run(seg,pol):
    tok=0;ms=0
    for a,c in seg:
        d,lev,syncs=pol(c)
        tok+=min(a,d)+1; ms+=V[d+1]+lev*L+OV+syncs*SYNC
    return tok/ms*1000, tok/len(seg)
def fixed(D): return lambda c:(D,D,0)
def rule(power=2.0,base_rate=None,most=8):
    # cuda_drafts-like: draft level j, keep it if run^power >= rate*row(j); continue if pays for next row+level
    def p(c):
        run=1;n=0;lev=0
        r=base_rate
        for j in range(1,most+1):
            lev=j; run*=c[j-1]
            row=V[j+1]-V[j]
            if j>1 and run**power < r*row: break
            n=j
            if j>=most or run**power < r*(V[j+2]-V[j+1]+L): break
        return n,lev,lev
    return p
def trim(D,power=2.0,r=0.1):
    # draft D levels, then verify the prefix maximizing expected tokens/ms
    def p(c):
        best=0;bv=-1;run=1;E=1
        for d in range(0,D+1):
            if d>0: run*=c[d-1]**power; E+=run
            v=E/(V[d+1]+D*L+OV)
            if v>bv: bv=v;best=d
        return best,D,1
    return p
pols={'fixed1':fixed(1),'fixed2':fixed(2),'fixed3':fixed(3),'fixed4':fixed(4),'fixed8':fixed(8)}
for r in (0.06,0.08,0.1,0.12):
  for pw in (1.0,2.0,3.0):
    pols[f'rule r{r} p{pw}']=rule(pw,r)
for D in (3,4,6,8):
    pols[f'trim{D}']=trim(D)
print('policy'.ljust(22),' '.join(k.rjust(14) for k in segs))
for n,p in pols.items():
    print(n.ljust(22),' '.join(('%.1f/%.2f'%run(s,p)).rjust(14) for s in segs.values()))
print()
class Adaptive:
    def __init__(s,power,zero=False,most=8,floorlev=False):
        s.tokens=2.0; s.ms=V[2]+L; s.power=power; s.zero=zero; s.most=most
    def __call__(s,c):
        r=s.tokens/s.ms; run=1;n=0;lev=0
        for j in range(1,s.most+1):
            lev=j; run*=c[j-1]
            row=V[j+1]-V[j]
            keep = (j==1 and not s.zero) or run**s.power >= r*row
            if not keep: break
            n=j
            if j>=s.most or run**s.power < r*(V[j+2]-V[j+1]+L): break
        s.last=(n,lev); return n,lev,lev
    def done(s,tokens,rows,levels):
        s.tokens+=0.125*(tokens-s.tokens); s.ms+=0.125*(V[rows]+levels*L-s.ms)
def runA(seg,pol):
    tok=0;ms=0
    for a,c in seg:
        d,lev,syncs=pol(c); t=min(a,d)+1
        tok+=t; ms+=V[d+1]+lev*L+OV+syncs*SYNC; pol.done(t,d+1,lev)
    return tok/ms*1000, tok/len(seg)
for pw in (1.0,1.5,2.0,3.0):
  for z in (False,True):
    print(f'adaptive p{pw} zero{z}'.ljust(22),' '.join(('%.1f/%.2f'%runA(s,Adaptive(pw,z))).rjust(14) for s in segs.values()))
