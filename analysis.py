import csv, math, random, os
from collections import Counter, defaultdict
SEED=42
random.seed(SEED)

DATA_PATH='data.csv'; OUT_DIR='results'; os.makedirs(OUT_DIR,exist_ok=True)

def read_data(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        rd=csv.DictReader(f,delimiter=';'); rows=list(rd); fields=rd.fieldnames
    feats=[c for c in fields if c!='Target']
    X=[[float(r[c]) for c in feats] for r in rows]
    y=[r['Target'] for r in rows]
    return feats,X,y

def standardize(X):
    n=len(X); d=len(X[0])
    mu=[sum(X[i][j] for i in range(n))/n for j in range(d)]
    sd=[]
    for j in range(d):
        v=sum((X[i][j]-mu[j])**2 for i in range(n))/n
        sd.append(math.sqrt(v) if v>1e-12 else 1.0)
    Z=[[(X[i][j]-mu[j])/sd[j] for j in range(d)] for i in range(n)]
    return Z

def stratified_split(y,test_ratio=0.2):
    rng=random.Random(SEED); cls=defaultdict(list)
    for i,t in enumerate(y): cls[t].append(i)
    tr=[]; te=[]
    for ids in cls.values():
        rng.shuffle(ids); cut=int(len(ids)*(1-test_ratio)); tr+=ids[:cut]; te+=ids[cut:]
    rng.shuffle(tr); rng.shuffle(te); return tr,te

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def sigmoid(z): return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))

def train_logreg(X,y,lr=0.05,epochs=200,l2=1e-3):
    n=len(X); d=len(X[0]); w=[0.0]*d; b=0.0
    for _ in range(epochs):
        gw=[0.0]*d; gb=0.0
        for xi,yi in zip(X,y):
            p=sigmoid(dot(w,xi)+b); e=p-yi
            for j in range(d): gw[j]+=e*xi[j]
            gb+=e
        for j in range(d): w[j]-=lr*(gw[j]/n+l2*w[j])
        b-=lr*gb/n
    return w,b

def pred_proba_lr(X,w,b): return [sigmoid(dot(w,xi)+b) for xi in X]

def knn_proba(Xt,yt,Xe,k=15):
    out=[]
    for x in Xe:
        ds=[]
        for a,b in zip(Xt,yt): ds.append((sum((x[j]-a[j])**2 for j in range(len(x))),b))
        ds.sort(key=lambda t:t[0]); out.append(sum(v for _,v in ds[:k])/k)
    return out

def confusion(y,p):
    tp=tn=fp=fn=0
    for a,b in zip(y,p):
        if a==1 and b==1: tp+=1
        elif a==0 and b==0: tn+=1
        elif a==0: fp+=1
        else: fn+=1
    return tp,tn,fp,fn

def metrics(y,prob):
    pred=[1 if p>=0.5 else 0 for p in prob]
    tp,tn,fp,fn=confusion(y,pred)
    acc=(tp+tn)/len(y); prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    auc=roc_auc(y,prob)[1]
    return dict(acc=acc,prec=prec,rec=rec,f1=f1,auc=auc,tp=tp,tn=tn,fp=fp,fn=fn)

def roc_auc(y,prob,steps=200):
    pts=[]
    for i in range(steps+1):
        t=i/steps; pred=[1 if p>=t else 0 for p in prob]
        tp,tn,fp,fn=confusion(y,pred); tpr=tp/(tp+fn) if tp+fn else 0; fpr=fp/(fp+tn) if fp+tn else 0
        pts.append((fpr,tpr))
    pts.sort(); auc=0
    for i in range(1,len(pts)):
        x1,y1=pts[i-1]; x2,y2=pts[i]; auc+=(x2-x1)*(y1+y2)/2
    return pts,auc

def pca2(X):
    n=len(X); d=len(X[0]); C=[[0.0]*d for _ in range(d)]
    for i in range(d):
        for j in range(i,d):
            v=sum(X[r][i]*X[r][j] for r in range(n))/n; C[i][j]=C[j][i]=v
    comps=[]; W=[row[:] for row in C]
    for _ in range(2):
        v=[random.random()-0.5 for _ in range(d)]
        for _ in range(100):
            nv=[sum(W[i][j]*v[j] for j in range(d)) for i in range(d)]
            norm=math.sqrt(sum(x*x for x in nv)) or 1; v=[x/norm for x in nv]
        lam=sum(v[i]*sum(W[i][j]*v[j] for j in range(d)) for i in range(d)); comps.append(v)
        for i in range(d):
            for j in range(d): W[i][j]-=lam*v[i]*v[j]
    return [[dot(x,comps[0]),dot(x,comps[1])] for x in X]

def kmeans(X,k=3,it=50):
    rng=random.Random(SEED); c=[X[rng.randrange(len(X))][:] for _ in range(k)]; lab=[0]*len(X)
    for _ in range(it):
        ch=0
        for i,x in enumerate(X):
            b=min(range(k), key=lambda j: sum((x[t]-c[j][t])**2 for t in range(len(x))))
            if b!=lab[i]: lab[i]=b; ch+=1
        for j in range(k):
            ids=[i for i,l in enumerate(lab) if l==j]
            if ids: c[j]=[sum(X[i][t] for i in ids)/len(ids) for t in range(len(X[0]))]
        if ch==0: break
    return lab

def kmedoids(X,k=3,it=12):
    rng=random.Random(SEED); med=rng.sample(range(len(X)),k); lab=[0]*len(X)
    for _ in range(it):
        for i,x in enumerate(X): lab[i]=min(range(k), key=lambda j: sum((x[t]-X[med[j]][t])**2 for t in range(len(x))))
        imp=False
        for j in range(k):
            ids=[i for i,l in enumerate(lab) if l==j]
            if not ids: continue
            best,bc=med[j],1e99
            for cand in ids[:80]:
                cc=sum(sum((X[i][t]-X[cand][t])**2 for t in range(len(X[0]))) for i in ids)
                if cc<bc: best,bc=cand,cc
            if best!=med[j]: med[j]=best; imp=True
        if not imp: break
    return lab

def silhouette(X,lab,cap=140):
    if len(X)>cap:
        idx=list(range(len(X))); random.Random(SEED).shuffle(idx); idx=idx[:cap]
        X=[X[i] for i in idx]; lab=[lab[i] for i in idx]
    cl=defaultdict(list)
    for i,l in enumerate(lab): cl[l].append(i)
    ss=[]
    for i,x in enumerate(X):
        own=lab[i]; ids=cl[own]
        a=sum(math.dist(x,X[j]) for j in ids if j!=i)/(len(ids)-1) if len(ids)>1 else 0
        b=min(sum(math.dist(x,X[j]) for j in ids2)/len(ids2) for c,ids2 in cl.items() if c!=own)
        ss.append((b-a)/max(a,b) if max(a,b)>0 else 0)
    return sum(ss)/len(ss)

def davies_bouldin(X,lab):
    cl=defaultdict(list)
    for i,l in enumerate(lab): cl[l].append(i)
    cent={k:[sum(X[i][j] for i in ids)/len(ids) for j in range(len(X[0]))] for k,ids in cl.items()}
    scat={k:sum(math.dist(X[i],cent[k]) for i in ids)/len(ids) for k,ids in cl.items()}
    ks=list(cl.keys()); vals=[]
    for i in ks:
        vals.append(max((scat[i]+scat[j])/math.dist(cent[i],cent[j]) for j in ks if j!=i and math.dist(cent[i],cent[j])>0))
    return sum(vals)/len(vals)

def write_csv(path,h,rows):
    with open(path,'w',newline='') as f: w=csv.writer(f); w.writerow(h); w.writerows(rows)

def main():
    feats,X,y=read_data(DATA_PATH); n=len(X); yb=[1 if t=='Dropout' else 0 for t in y]
    Z=standardize(X); tr,te=stratified_split(y)
    Xtr=[Z[i] for i in tr]; Xte=[Z[i] for i in te]; ytr=[yb[i] for i in tr]; yte=[yb[i] for i in te]

    w,b=train_logreg(Xtr,ytr); ptr_lr=pred_proba_lr(Xtr,w,b); pte_lr=pred_proba_lr(Xte,w,b); pall_lr=pred_proba_lr(Z,w,b)
    sub=list(range(len(Xtr))); random.Random(SEED).shuffle(sub); sub=sub[:900]
    pte_knn=knn_proba([Xtr[i] for i in sub],[ytr[i] for i in sub],Xte,k=17)

    E={("LR","train"):metrics(ytr,ptr_lr),("LR","test"):metrics(yte,pte_lr),("LR","all"):metrics(yb,pall_lr),
       ("KNN","test"):metrics(yte,pte_knn)}

    cv=[(1e-4,0.0),(1e-3,0.0),(1e-2,0.0),(1e-1,0.0)]

    # clustering/visualization on subset
    sidx=list(range(n)); random.Random(SEED).shuffle(sidx); sidx=sidx[:320]
    Xs=[Z[i] for i in sidx]; ys=[y[i] for i in sidx]
    km=kmeans(Xs,3); md=kmedoids(Xs,3)
    skm,dkm=silhouette(Xs,km),davies_bouldin(Xs,km)
    smd,dmd=silhouette(Xs,md),davies_bouldin(Xs,md)
    emb=pca2(Xs)

    write_csv(f'{OUT_DIR}/model_metrics.csv',['model','split','accuracy','precision','recall','f1','auc','tp','tn','fp','fn'],
      [[m,s,f"{v['acc']:.4f}",f"{v['prec']:.4f}",f"{v['rec']:.4f}",f"{v['f1']:.4f}",f"{v['auc']:.4f}",v['tp'],v['tn'],v['fp'],v['fn']] for (m,s),v in E.items()])
    write_csv(f'{OUT_DIR}/clustering_metrics.csv',['algorithm','silhouette','davies_bouldin'],[['kmeans',f'{skm:.4f}',f'{dkm:.4f}'],['kmedoids',f'{smd:.4f}',f'{dmd:.4f}']])
    write_csv(f'{OUT_DIR}/embedding_points.csv',['x','y','label','kmeans','kmedoids'],[[f'{emb[i][0]:.6f}',f'{emb[i][1]:.6f}',ys[i],km[i],md[i]] for i in range(len(emb))])
    for name,prob in [('lr',pte_lr),('knn',pte_knn)]:
        pts,_=roc_auc(yte,prob); write_csv(f'{OUT_DIR}/roc_{name}.csv',['fpr','tpr'],[[f'{a:.6f}',f'{b:.6f}'] for a,b in pts])
    with open(f'{OUT_DIR}/summary.txt','w') as f:
        f.write(f'samples={n},features={len(feats)},class_counts={dict(Counter(y))}\n')
        f.write(f'cluster kmeans sil={skm:.4f} db={dkm:.4f}; kmedoids sil={smd:.4f} db={dmd:.4f}\n')
        f.write(f'cv_l2={cv}\n')
        for k,v in E.items(): f.write(f'{k}:{v}\n')
    print('done')

if __name__=='__main__': main()
