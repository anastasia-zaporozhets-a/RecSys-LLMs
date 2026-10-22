#!/usr/bin/env python3
from __future__ import annotations
import csv, os, platform, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

EXAM = Path(__file__).resolve().parents[1]
DATA = EXAM / "data"
OUT = EXAM / "out"
OUT.mkdir(exist_ok=True)
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
REPORT_LINES = []

def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT_LINES.append(str(msg))

def hr(title):
    log("")
    log("=" * 100)
    log(title)
    log("=" * 100)

def load_schedules(blocks):
    sched = {}
    for b in blocks:
        path = DATA / f"export_arh_{b}-final.csv"
        t0 = time.perf_counter()
        df = pd.read_csv(path, dtype={"tv_show_id": "int64", "channel_id": "int64", "duration": "int64"}, low_memory=False)
        df = df[df["tv_show_id"] != 0].copy()
        df["start_local"] = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M:%S", errors="raise")
        sched[b] = df
        log(f"  loaded schedule {b}: {len(df):,} labelled rows in {time.perf_counter() - t0:.2f}s")
    return sched

def get_pool_and_airtime(sched_df):
    pool = set(int(x) for x in sched_df["tv_show_id"].unique())
    airtime = sched_df.groupby("tv_show_id")["duration"].sum().astype(np.int64)
    return pool, airtime

def load_interactions():
    t0 = time.perf_counter()
    df = pd.read_csv(OUT / "interactions.csv", dtype={"user_id": str, "tv_show_id": str, "block": str})
    df["tv_show_id"] = df["tv_show_id"].astype(np.int64)
    df["overlap_seconds"] = df["overlap_seconds"].astype(np.int64)
    df["airings"] = df["airings"].astype(np.int64)
    log(f"  loaded interactions: {len(df):,} rows in {time.perf_counter() - t0:.2f}s")
    return df

def load_programmes():
    t0 = time.perf_counter()
    df = pd.read_csv(OUT / "programmes.csv", dtype={"tv_show_id": "int64"})
    log(f"  loaded programmes: {len(df):,} rows in {time.perf_counter() - t0:.2f}s")
    return df

def get_user_train(inter_df, train_blocks):
    train = inter_df[inter_df["block"].isin(train_blocks)].copy()
    user_train_sum = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    user_train_sum = user_train_sum.sort_values(["user_id", "overlap_seconds", "tv_show_id"], ascending=[True, False, True])
    return user_train_sum

def get_eval_targets(inter_df, eval_block):
    eval_df = inter_df[inter_df["block"] == eval_block].copy()
    g = eval_df.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    g = g.sort_values(["user_id", "overlap_seconds", "tv_show_id"], ascending=[True, False, True])
    targets_top5 = {}
    targets_all = {}
    for uid, grp in g.groupby("user_id"):
        targets_all[uid] = set(grp["tv_show_id"].tolist())
        targets_top5[uid] = grp["tv_show_id"].tolist()[:5]
    return targets_top5, targets_all

def ap_at_k(predicted, actual_set, k=5):
    if len(actual_set) == 0: return 0.0
    score = 0.0; num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual_set), k)

def map_at_k(pred_lists, targets, k=5):
    users = [u for u in pred_lists if u in targets and len(targets[u]) >= 1]
    if not users: return 0.0, 0
    s = 0.0
    for u in users:
        s += ap_at_k(pred_lists[u], targets[u], k)
    return s / len(users), len(users)

def build_popularity(train_sum, pool):
    users_per_item = train_sum.groupby("tv_show_id")["user_id"].nunique()
    pop = users_per_item[users_per_item.index.isin(pool)].copy()
    pop = pop.sort_values(ascending=False)
    return pop

def make_recs(method_scores, pool, popularity_ordered, user_train_set, cold_start_users, k=5):
    recs = {}
    fallback_count = 0
    all_uids = set(method_scores.keys()) | set(cold_start_users)
    for uid in all_uids:
        if uid in cold_start_users or uid not in user_train_set or len(user_train_set[uid]) == 0:
            lst = []
            for it in popularity_ordered.index:
                if int(it) in pool:
                    lst.append(int(it))
                if len(lst) == k: break
            recs[uid] = lst[:k]
            fallback_count += 1
            continue
        ranked = []
        seen = set()
        for it, sc in method_scores.get(uid, []):
            it = int(it)
            if it in seen: continue
            seen.add(it)
            ranked.append(it)
            if len(ranked) == k: break
        if len(ranked) < k:
            for it in popularity_ordered.index:
                it = int(it)
                if it in seen: continue
                seen.add(it)
                ranked.append(it)
                if len(ranked) == k: break
        recs[uid] = ranked[:k]
    return recs, fallback_count

def ppr_batched(eval_users_val, user2idx, item2idx, user_list_ppr, item_list_ppr,
                user_train_list_val, user_train_set_val, pool_val, alpha, chunk_size=500, max_iter=30, tol=1e-6):
    t0 = time.perf_counter()
    n_u = len(user_list_ppr); n_i = len(item_list_ppr); n = n_u + n_i
    rows=[]; cols=[]; data=[]
    for uid, items in user_train_list_val.items():
        if uid not in user2idx: continue
        for it, sec in items:
            if it not in item2idx: continue
            w = np.log1p(float(sec)/60.0)
            u=user2idx[uid]; i=item2idx[it]
            rows.append(u); cols.append(n_u+i); data.append(w)
            rows.append(n_u+i); cols.append(u); data.append(w)
    A = sparse.coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
    deg = np.array(A.sum(axis=1)).ravel(); deg[deg==0]=1.0
    Dinv = sparse.diags(1.0/deg)
    P = (Dinv @ A).tocsr()
    PT = P.T.tocsr()
    eval_list = [uid for uid in eval_users_val if uid in user2idx and uid in user_train_set_val and len(user_train_set_val[uid])>0]
    log(f"  PPR alpha={alpha}: n={n} (u={n_u},i={n_i}), eval_with_train={len(eval_list)}, chunks={chunk_size}, max_iter={max_iter}")
    R_items_all = {}
    for cstart in range(0,len(eval_list),chunk_size):
        cend = min(cstart+chunk_size,len(eval_list))
        chunk = eval_list[cstart:cend]
        tchunk=time.perf_counter()
        m=len(chunk)
        V = np.zeros((n,m))
        for j,uid in enumerate(chunk):
            uidx=user2idx[uid]
            s=0.0
            for it,sec in user_train_list_val[uid]:
                if it in item2idx: s += np.log1p(float(sec)/60.0)
            if s==0: V[uidx,j]=1.0
            else:
                for it,sec in user_train_list_val[uid]:
                    if it in item2idx: V[n_u+item2idx[it],j] += np.log1p(float(sec)/60.0)/s
        V = (1.0-alpha)*V
        R = V.copy()
        for it in range(max_iter):
            R_new = alpha*(PT@R)+V
            diff = np.max(np.sum(np.abs(R_new-R),axis=0))
            R = R_new
            if diff < tol:
                log(f"    chunk {cstart//chunk_size+1} ({len(chunk)} users) conv at iter {it+1}, diff={diff:.2e}, {time.perf_counter()-tchunk:.1f}s")
                break
        else:
            log(f"    chunk {cstart//chunk_size+1} ({len(chunk)} users) done {max_iter} iters, {time.perf_counter()-tchunk:.1f}s")
        R_items = R[n_u:n_u+n_i,:]
        for j,uid in enumerate(chunk):
            scores = R_items[:,j]
            scored=[]
            for k,it in enumerate(item_list_ppr):
                if int(it) not in pool_val: continue
                scored.append((int(it), float(scores[k])))
            scored.sort(key=lambda x:(-x[1],x[0]))
            R_items_all[uid]=scored
    log(f"  PPR alpha={alpha} total: {time.perf_counter()-t0:.1f}s")
    return R_items_all

def main():
    hr("STEP 2: FULL EVALUATION (STRING USER IDS)")
    log(f"Environment: Python {sys.version.split()[0]} {platform.platform()}")
    log(f"numpy {np.__version__} pandas {pd.__version__}")
    log(f"Random seed: {RANDOM_SEED}")
    inter = load_interactions()
    progs = load_programmes()
    sched_all = load_schedules(["11-20","21-30","31-42"])
    train_blocks_val=["11-20"]; eval_block_val="21-30"
    train_sum_val = get_user_train(inter,train_blocks_val)
    pool_val, airtime_val = get_pool_and_airtime(sched_all[eval_block_val]); pool_val=set(int(x) for x in pool_val)
    targets_top5_val, targets_all_val = get_eval_targets(inter,eval_block_val)
    n_eval = len([u for u in targets_all_val if len(targets_all_val[u])>=1])
    log(f"Validation: train {train_blocks_val} eval {eval_block_val}, pool {len(pool_val)}, eval_users {n_eval}")
    assert n_eval==4799
    user_train_set_val=defaultdict(set); user_train_list_val=defaultdict(list)
    for _,row in train_sum_val.iterrows():
        uid=row["user_id"]; it=int(row["tv_show_id"])
        user_train_set_val[uid].add(it); user_train_list_val[uid].append((it,float(row["overlap_seconds"])))
    for uid in user_train_list_val: user_train_list_val[uid].sort(key=lambda x:(-x[1],x[0]))
    log(f"training_users {len(user_train_set_val)}")
    assert len(user_train_set_val)==4834
    cold_start_users_val = set(targets_all_val.keys())-set(user_train_set_val.keys())
    log(f"cold_start {len(cold_start_users_val)}")
    pop_val = build_popularity(train_sum_val,pool_val); popularity_ordered_val=pop_val
    eval_users_val=list(targets_all_val.keys())
    sub=pd.read_csv(DATA/"submission.csv",dtype={"user_id":str}); log(f"submission {len(sub)}")
    
    # B1
    hr("B1")
    t0=time.perf_counter()
    pop_recs_val2={}
    for uid in eval_users_val:
        lst=[]
        for it in popularity_ordered_val.index:
            it=int(it)
            if it not in pool_val: continue
            lst.append(it)
            if len(lst)==5: break
        pop_recs_val2[uid]=lst
    t_b1=time.perf_counter()-t0
    map_top5_b1,_ = map_at_k(pop_recs_val2,targets_top5_val,5)
    map_all_b1,_ = map_at_k(pop_recs_val2,targets_all_val,5)
    log(f"B1 MAP@5 top5 {map_top5_b1:.6f} all {map_all_b1:.6f} rt {t_b1:.3f}s")
    
    # B2
    hr("B2")
    t0=time.perf_counter()
    hist_recs_val={}
    for uid in eval_users_val:
        lst=[]
        for it,sc in user_train_list_val.get(uid,[]):
            it=int(it)
            if it not in pool_val: continue
            if it in lst: continue
            lst.append(it)
            if len(lst)==5: break
        if len(lst)<5:
            for it in popularity_ordered_val.index:
                it=int(it)
                if it in lst: continue
                if it not in pool_val: continue
                lst.append(it)
                if len(lst)==5: break
        hist_recs_val[uid]=lst
    t_b2=time.perf_counter()-t0
    map_top5_b2,_ = map_at_k(hist_recs_val,targets_top5_val,5)
    map_all_b2,_ = map_at_k(hist_recs_val,targets_all_val,5)
    log(f"B2 MAP@5 top5 {map_top5_b2:.6f} all {map_all_b2:.6f} rt {t_b2:.3f}s")
    
    # M1
    hr("M1")
    all_train_users=list(user_train_set_val.keys()); train_items=list(set(train_sum_val["tv_show_id"].astype(int).tolist()))
    user2idx={u:i for i,u in enumerate(all_train_users)}; item2idx={it:i for i,it in enumerate(train_items)}
    alphas=[0.3,0.5,0.7,0.85]
    ppr_results={}
    for alpha in alphas:
        t_est=time.perf_counter()
        recs_ppr_scores = ppr_batched(eval_users_val,user2idx,item2idx,all_train_users,train_items,user_train_list_val,user_train_set_val,pool_val,alpha)
        method_scores_ppr={uid: recs_ppr_scores.get(uid,[]) for uid in eval_users_val}
        final_recs_ppr,fb_ppr = make_recs(method_scores_ppr,pool_val,popularity_ordered_val,user_train_set_val,cold_start_users_val,5)
        map_top5_ppr,_=map_at_k(final_recs_ppr,targets_top5_val,5)
        map_all_ppr,_=map_at_k(final_recs_ppr,targets_all_val,5)
        ppr_results[alpha]=(map_top5_ppr,map_all_ppr,time.perf_counter()-t_est,fb_ppr,final_recs_ppr)
        log(f"alpha={alpha}: top5 {map_top5_ppr:.6f} all {map_all_ppr:.6f} rt {ppr_results[alpha][2]:.1f}s fb {fb_ppr}")
    best_alpha=max(ppr_results.keys(),key=lambda a:ppr_results[a][0])
    log(f"best_alpha {best_alpha}")
    
    # M2
    hr("M2")
    t0=time.perf_counter()
    def norm_title(t):
        t=str(t).lower()
        t=re.sub(r"[,.:;!?()\[\]\"']", " ", t)
        t=re.sub(r"\b(sezon|season|s\d+|эп\.?|серия|series)\b.*$", " ", t)
        t=re.sub(r"\d+\s*эп.*$", " ", t)
        t=re.sub(r"\s+"," ",t).strip()
        return t
    prog_info={}
    for _,row in progs.iterrows():
        pid=int(row["tv_show_id"])
        prog_info[pid]={"title":row["title"],"cat":str(row["category"]) if pd.notna(row["category"]) else "","gens":str(row["genres"]) if pd.notna(row["genres"]) else ""}
    all_pids=list(prog_info.keys())
    docs=[]
    for pid in all_pids:
        p=prog_info[pid]; nt=norm_title(p["title"])
        doc=f"CAT_{p['cat']} " + " ".join(f"GEN_{g.strip()}" for g in p["gens"].split(";") if g.strip()) + " " + nt
        docs.append(doc)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec=TfidfVectorizer(max_features=5000,min_df=2); X=vec.fit_transform(docs)
    pid_to_idx={pid:i for i,pid in enumerate(all_pids)}
    user_profile={}
    for uid,items in user_train_list_val.items():
        total_w=0.0; prof=None
        for it,sec in items:
            if it not in pid_to_idx: continue
            w=float(sec); total_w+=w
            if prof is None: prof=X[pid_to_idx[it]].toarray().ravel()*w
            else: prof+=X[pid_to_idx[it]].toarray().ravel()*w
        if prof is not None and total_w>0:
            nrm=np.linalg.norm(prof)
            if nrm>0: prof=prof/nrm
            user_profile[uid]=prof
    item_vecs_norm={}
    for i,pid in enumerate(all_pids):
        v=X[i].toarray().ravel(); nrm=np.linalg.norm(v)
        if nrm>0: v=v/nrm
        item_vecs_norm[pid]=v
    recs_cb_alone={}
    for uid in eval_users_val:
        if uid not in user_profile: recs_cb_alone[uid]=[]; continue
        up=user_profile[uid]
        scored=[]
        for pid in pool_val:
            if pid not in item_vecs_norm: continue
            s=float(np.dot(up,item_vecs_norm[pid]))
            scored.append((pid,s))
        scored.sort(key=lambda x:(-x[1],x[0])); recs_cb_alone[uid]=scored
    method_scores_cb={uid:recs_cb_alone[uid] for uid in eval_users_val}
    final_recs_cb,fb_cb = make_recs(method_scores_cb,pool_val,popularity_ordered_val,user_train_set_val,cold_start_users_val,5)
    t_cb=time.perf_counter()-t0
    map_top5_cb,_=map_at_k(final_recs_cb,targets_top5_val,5)
    map_all_cb,_=map_at_k(final_recs_cb,targets_all_val,5)
    t0b=time.perf_counter()
    recs_cb_tie={}
    for uid in eval_users_val:
        if uid not in user_profile: recs_cb_tie[uid]=[]; continue
        up=user_profile[uid]; hist_set=user_train_set_val.get(uid,set())
        scored=[]
        for pid in pool_val:
            if pid not in item_vecs_norm: continue
            s=float(np.dot(up,item_vecs_norm[pid]))
            scored.append((pid,(s,1.0 if pid in hist_set else 0.0)))
        scored.sort(key=lambda x:(-x[1][0],-x[1][1],x[0]))
        recs_cb_tie[uid]=[(pid,sc[0]) for pid,sc in scored]
    method_scores_cb_tie={uid:recs_cb_tie[uid] for uid in eval_users_val}
    final_recs_cb_tie,fb_cb_tie = make_recs(method_scores_cb_tie,pool_val,popularity_ordered_val,user_train_set_val,cold_start_users_val,5)
    t_cb_tie=time.perf_counter()-t0b
    map_top5_cb_tie,_=map_at_k(final_recs_cb_tie,targets_top5_val,5)
    map_all_cb_tie,_=map_at_k(final_recs_cb_tie,targets_all_val,5)
    log(f"M2 alone top5 {map_top5_cb:.6f} all {map_all_cb:.6f} rt {t_cb:.1f}s fb {fb_cb}")
    log(f"M2+tie top5 {map_top5_cb_tie:.6f} all {map_all_cb_tie:.6f} rt {t_cb_tie:.1f}s fb {fb_cb_tie}")
    
    hr("TABLE")
    log("method | params | MAP@5 (top-5 target) | MAP@5 (all-watched) | runtime")
    log(f"B1_popularity | distinct_users_watched_in_train, restricted_to_pool | {map_top5_b1:.6f} | {map_all_b1:.6f} | {t_b1:.3f}s")
    log(f"B2_personal_history | user_train_by_overlap_sec, restricted_to_pool | {map_top5_b2:.6f} | {map_all_b2:.6f} | {t_b2:.3f}s")
    for alpha in alphas:
        r=ppr_results[alpha]
        log(f"M1_PPR | alpha={alpha}, log1p(sec/60), tol=1e-6, max_iter=30, batched | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.1f}s")
    log(f"M2_content | cat+3genres+TFIDF_title(norm), user_profile_sec_weighted_mean, cosine | {map_top5_cb:.6f} | {map_all_cb:.6f} | {t_cb:.1f}s")
    log(f"M2_content+hist_tiebreak | same + prefer_history_on_tie | {map_top5_cb_tie:.6f} | {map_all_cb_tie:.6f} | {t_cb_tie:.1f}s")
    br=ppr_results[best_alpha]
    log(f"selected PPR alpha={best_alpha} fallback={br[3]}")
    
    def save_val(name,recs):
        path=OUT/f"val_{name}.csv"
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['user_id']+[str(i) for i in range(1,6)])
            for uid in sorted(recs.keys()):
                lst=recs[uid][:5]; lst=list(dict.fromkeys(lst))
                while len(lst)<5:
                    for it in popularity_ordered_val.index:
                        it=int(it)
                        if it in lst or it not in pool_val: continue
                        lst.append(it)
                        if len(lst)==5: break
                w.writerow([uid]+lst[:5])
        log(f"wrote {path}")
    save_val('popularity',pop_recs_val2)
    save_val('personal_history',hist_recs_val)
    for alpha in alphas: save_val(f'ppr_alpha{alpha}',ppr_results[alpha][4])
    save_val('content_based',final_recs_cb)
    save_val('content_based_hist_tiebreak',final_recs_cb_tie)
    (OUT/"step2_report.md").write_text("\n".join(REPORT_LINES)+"\n",encoding='utf-8')
    print("\nDONE")

if __name__=='__main__': main()
