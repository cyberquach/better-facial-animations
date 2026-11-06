"""Local calibration regression, skin-only; does not install or alter any mod.
Invoked by tools/face_audit/run.py with a frozen source snapshot.
"""
from pathlib import Path
import os,sys,json,struct,re,ctypes as C,time,traceback,argparse,collections
import numpy as np
from assets import Assets,ROOT,PROFILE,key,sha,named_shapes
from checks import can_skip_speech
import tri
from bsa import BSA

F=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
class Mapper:
    def __init__(self,name,model=3):
        self.lib=C.CDLL(str(ROOT/(name+'.so')))
        self.lib.init.argtypes=[C.c_char_p];self.lib.init.restype=C.c_int
        self.lib.fit.argtypes=[C.c_int,C.c_int,F,F,F,C.c_char_p];self.lib.fit.restype=C.c_void_p
        self.lib.fingerprint.argtypes=[C.c_void_p];self.lib.fingerprint.restype=C.c_uint64
        self.lib.evaluate.argtypes=[C.c_void_p,C.c_int,C.c_int,F,F];self.lib.evaluate.restype=C.c_int
        self.lib.evaluate_mode.argtypes=[C.c_void_p,C.c_int,C.c_int,F,F,C.c_int];self.lib.evaluate_mode.restype=C.c_int
        self.lib.has_extended_controls.argtypes=[C.c_void_p]
        self.lib.has_extended_controls.restype=C.c_bool
        self.lib.evaluate_controls.argtypes=[C.c_void_p,C.c_int,C.c_int,F,F,C.c_int,C.c_float,C.c_float,C.c_int]
        self.lib.has_repair_state.argtypes=[C.c_void_p]
        self.lib.has_repair_state.restype=C.c_bool
        self.lib.release.argtypes=[C.c_void_p]
        self.lib.repair_mask.argtypes=[C.c_void_p,np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')]
        self.lib.expression_fingerprint.argtypes=[C.c_void_p]
        self.lib.expression_fingerprint.restype=C.c_uint64
        assert self.lib.init(str(ROOT/'models'/('A2FDirect3.bin' if model==3 else 'A2FDirect.bin')).encode())
    def fit(self,v,m,female):
        stats=np.zeros(22,np.float32);why=C.create_string_buffer(1024)
        h=self.lib.fit(female,len(v),v,m,stats,why)
        r=dict(accepted=bool(h),repaired=int(stats[21]),ignored=int(stats[1]),scale=float(stats[2]),lip_distance=float(stats[3]),lip_counts=stats[4:6].astype(int).tolist(),residual=stats[7:15].tolist(),auto_scale=stats[15:21].tolist(),reason=why.value.decode())
        if h:
            r['fingerprint']='%016x'%self.lib.fingerprint(h)
            r['has_repair_state']=bool(self.lib.has_repair_state(h))
            r['extended_controls']=bool(self.lib.has_extended_controls(h))
        return h,r
    def evaluate(self,h,coef,nv,limited=False):
        out=np.empty((len(coef),nv,3),np.float32)
        assert self.lib.evaluate_mode(h,len(coef),coef.shape[1],coef,out,int(limited))
        return out

def stimulus(path,full=False):
    b=Path(path).read_bytes();nf,dim,fps=struct.unpack_from('<IIf',b)
    c=np.frombuffer(b,np.float32,offset=12).reshape(nf,dim)
    # Evenly spaced speech plus highest coefficient norm frames; deterministic.
    ix=list(range(nf)) if full else sorted(set(np.linspace(0,nf-1,32,dtype=int).tolist()+np.argsort(np.linalg.norm(c,axis=1))[-8:].tolist()))
    return np.ascontiguousarray(c[ix]),dict(path=path,sha256=sha(b),total_frames=nf,selected_frames=ix,fps=fps,dim=dim)

def landmarks(t):
    """Independent native-TRI lip-pair candidates, never BFA lip labels.
    Front, central, close opposite-motion pairs. Heuristic screening, not ground truth.
    """
    v=t['verts'];d=t['morphs'].get('BigAah')
    if d is None:return [],'no BigAah'
    extent=np.ptp(v,axis=0);diag=float(np.linalg.norm(extent));drop=-d[:,2];peak=drop.max()
    if peak<1e-5:return [],'no native opening'
    front=np.quantile(v[:,1],.80);width=extent[0];pairs=[]
    for fraction in [-.035,0,.035]:
        x=(v[:,0].max()+v[:,0].min())/2+fraction*width
        near=abs(v[:,0]-x)<width*.022
        lower=np.flatnonzero((drop>.35*peak)&near&(v[:,1]>front))
        upper=np.flatnonzero((drop<.15*peak)&near&(v[:,1]>front))
        candidates=[]
        for u in upper:
            for l in lower:
                dist=float(np.linalg.norm(v[u]-v[l]));dz=v[u,2]-v[l,2];gain=float(drop[l]-drop[u])
                if 0<dz<.05*diag and dist<.05*diag:
                    # Prioritize the visible front surface over internal lip copies.
                    score=min(v[u,1],v[l,1])-.3*dist-.5*(abs(v[u,0]-x)+abs(v[l,0]-x))
                    candidates.append((score,int(u),int(l),gain))
        if candidates:
            _,u,l,gain=max(candidates);pairs.append([u,l,gain])
    return pairs,'native front-surface heuristic; requires visual review'

def upper_surface_edges(v,m,tris,pairs):
    """Independent TRI upper-mouth surface screen; not mapper repair labels."""
    if not pairs:return np.empty((0,2),int)
    centre=np.median(v[np.array(pairs,int)[:,0]],axis=0)
    extent=np.ptp(v,axis=0);diag=float(np.linalg.norm(extent))
    peak=float(np.max(-m[4,:,2]))
    upper=(abs(v[:,0]-centre[0])<.16*extent[0]) & (v[:,1]>np.quantile(v[:,1],.80)) & (v[:,2]>centre[2]-.02*diag) & (v[:,2]<centre[2]+.05*diag) & (-m[4,:,2]<.12*peak)
    edges=np.unique(np.sort(np.concatenate([tris[:,[0,1]],tris[:,[1,2]],tris[:,[2,0]]]),axis=1),axis=0)
    rest=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    return edges[upper[edges].all(1)&(rest>1e-4*diag)]

def metrics(v,out,pairs,edges=None):
    if not np.isfinite(out).all():return dict(finite=False)
    diag=float(np.linalg.norm(np.ptp(v,axis=0)))
    r=dict(finite=True,max_displacement=float(np.linalg.norm(out,axis=2).max()),head_diagonal=diag)
    if edges is not None and len(edges):
        u,l=edges[:,0],edges[:,1]
        rest=v[u]-v[l];length=np.linalg.norm(rest,axis=1)
        strain=np.max(np.abs(np.linalg.norm(rest+out[:,u]-out[:,l],axis=2)/length-1),axis=0)
        r['upper_surface']={'edges':len(edges),'p95_strain':float(np.quantile(strain,.95)),'over_double_length':int(np.sum(strain>1)),'max_strain':float(strain.max())}
    if not pairs:return r
    ix=np.array([[x[0],x[1]] for x in pairs],int);u,l=ix[:,0],ix[:,1]
    native=np.array([x[2] for x in pairs])
    delta=out[:,u,2]-out[:,l,2];rest=v[u,2]-v[l,2];gap=rest+delta
    moving=np.linalg.norm(out[:,u]-out[:,l],axis=2)
    total=np.linalg.norm(out[:,u],axis=2)+np.linalg.norm(out[:,l],axis=2)
    peaks=np.max(delta,axis=0)
    r.update(pair_peak_increase=peaks.tolist(),opening_ratio=float(np.median(peaks/native)),peak_gap=float(gap.max()),minimum_gap=float(gap.min()),rest_gaps=rest.tolist(),common_motion_fraction=float(np.sum(total-moving)/max(np.sum(total),1e-8)),peak_frame=int(np.argmax(np.mean(delta,axis=1))))
    return r

def penetration_worsening(v, before, after, pairs):
    if not pairs: return 0.0
    ix=np.asarray(pairs,dtype=int);u,l=ix[:,0],ix[:,1]
    rest=v[u,2]-v[l,2]
    old=rest+before[:,u,2]-before[:,l,2]
    new=rest+after[:,u,2]-after[:,l,2]
    return float(np.maximum(0,np.minimum(0,old)-new).max())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--out',default='results.jsonl');ap.add_argument('--only');ap.add_argument('--full',action='store_true');ap.add_argument('--installed',action='store_true');ap.add_argument('--model',type=int,choices=[2,3],default=3);ap.add_argument('--strength-sweep',action='store_true');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
    a=Assets(cached=True);old=Mapper('old',args.model);new=Mapper('new',args.model)
    speech={};stimuli={}
    clips=[(sex, json.loads(Path(os.environ['BFA_AUDIT_STIMULI']).read_text())[str(args.model)][str(int(sex))]) for sex in [True,False]]
    for sex,path in clips:speech[sex],stimuli[str(sex)]=stimulus(path,args.full)
    names=re.findall(r'\{\d, \d+, "([^"]+)"\}',Path(os.environ['BFA_AUDIT_SOURCE']).read_text().split('};',1)[0]);assert len(names)==36
    receipt=dict(scope='Active-profile candidate FaceGen skin fitting and sampled speech, not game certification. Unreferenced assets also screened. Other shapes, runtime substitution, audio inference and occlusion are outside this probe.',profile=PROFILE.name,profile_hashes={p:sha((PROFILE/p).read_bytes()) for p in ['modlist.txt','plugins.txt','loadorder.txt','archives.txt']},source_hashes={p:sha(Path(p).read_bytes()) for p in [os.environ['BFA_AUDIT_SOURCE'],str(ROOT/'include/a2f/animation/direct.h'),os.environ['BFA_AUDIT_BASELINE'],str(ROOT/'models/A2FDirect3.bin')]},stimuli=stimuli,archive_order=a.archive_order,errors=a.errors)
    receipt.update(model=args.model,full_speech=args.full,installed_variants=args.installed)
    receipt['source_hashes']['dist/A2FDirect.bin']=sha((ROOT/'models/A2FDirect.bin').read_bytes())
    (ROOT/(args.out+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
    cache={};results={};start=time.time();counts=collections.Counter();archives={}
    installed=json.loads((ROOT/'installed.json').read_text()) if args.installed else None
    tasks=installed['tasks'] if installed else [{'path':p} for p in a.head_paths()]
    if args.only:tasks=[t for t in tasks if args.only.lower() in t['path']]
    if args.limit:tasks=tasks[:args.limit]
    tasks=tasks[args.shard::args.shards]
    def read_source(src):
        if 'file' in src:return Path(src['file']).read_bytes()
        if src['archive'] not in archives:archives[src['archive']]=BSA(src['archive'])
        return archives[src['archive']].read(src['path'])
    output=ROOT/args.out
    with output.open('w') as log:
        for index,task in enumerate(tasks):
            path=task['path'];r={'path':path};h0=h1=None
            local=installed['roots'][task['root']] if installed else None
            if local:r.update(variant_source=task['source'],variant_root=task['root'],mod_active=local['active'],scope='source_variant_over_active_profile')
            try:
                plugin,fn=path.split('\\')[-2:];fid=int(Path(fn).stem,16)&0xffffff;ident=plugin+':%06x'%fid
                npc=a.npcs.get(ident);r.update(npc_id=ident,npc=npc)
                data,src=(read_source(task['source']),task['source']) if local else a.read(path)
                r.update(mesh_source=src,mesh_sha256=sha(data))
                shapes=named_shapes(data);found=[]
                for s in shapes:
                    parts=(local['parts'].get(s['name'].lower()) if local else None) or a.part_names.get(s['name'].lower(),[])
                    face=[p for p in parts if p['type']==1]
                    if npc and not local and len(face)>1:
                        explicit=[p for p in face if p.get('id') in npc['parts']]
                        if explicit:face=explicit
                    if face:found.append((s,face))
                    elif not parts and 'head' in s['name'].lower() and not any(x in s['name'].lower() for x in ['brow','beard']):found.append((s,[]))
                if len(found)!=1:raise ValueError('skin_shape_count:'+str(len(found)))
                s,parts=found[0];r['shape']=s['name'];r['nv']=len(s['verts'])
                tris={p['tri'] for p in parts if p['tri']}
                if len(tris)!=1:raise ValueError('expression_tri_candidates:'+str(sorted(tris)))
                tp=next(iter(tris))
                cachekey=(task['root']+'|'+tp) if local else tp
                if cachekey not in cache:
                    candidates=local['tris'].get(tp,[]) if local else []
                    if candidates:
                        loose=[c for c in candidates if 'file' in c];candidates=loose or candidates
                        contents=[(read_source(c),c) for c in candidates]
                        if len({sha(b) for b,c in contents})!=1:raise ValueError('ambiguous_local_expression_tri')
                        b,ts=contents[0]
                    else:b,ts=a.read(tp)
                    dest=ROOT/'tri-cache'/ (sha(b)+'.tri');dest.parent.mkdir(exist_ok=True)
                    if not dest.exists():
                        temp=dest.with_suffix('.'+str(os.getpid())+'.tmp');temp.write_bytes(b);temp.replace(dest)
                    t=tri.read(dest);pairs,method=landmarks(t)
                    morph=np.array([t['morphs'].get(n,np.zeros_like(t['verts'])) for n in names],np.float32)
                    cache[cachekey]=(t,morph,pairs,dict(path=tp,source=ts,sha256=sha(b),missing_morphs=[n for n in names if n not in t['morphs']],landmarks=pairs,landmark_method=method))
                t,m,pairs,info=cache[cachekey];r['tri']=info
                v=np.ascontiguousarray(s['verts'],np.float32)
                if len(v)!=len(t['verts']):raise ValueError('vertex_mismatch: %d vs %d'%(len(v),len(t['verts'])))
                if not np.isfinite(v).all() or not np.isfinite(m).all():raise ValueError('nonfinite_input')
                # Canonical sex from winning NPC; unresolved asset sex inferred explicitly.
                female=npc['female'] if npc else 'female' in s['name'].lower()
                r['female']=female;r['sex_source']='NPC' if npc else 'shape_name_heuristic'
                digest=sha(v.tobytes()+m.tobytes()+bytes([female]));r['input_sha256']=digest
                if digest in results:
                    r.update(results[digest]);r['duplicate_of']=results[digest]['canonical_path']
                else:
                    edges=upper_surface_edges(v,m,t['tris'],pairs)
                    h0,r0=old.fit(v,m,female);h1,r1=new.fit(v,m,female)
                    r.update(old=r0,new=r1,canonical_path=path)
                    r['mapping_identical']=bool(h0 and h1 and can_skip_speech(r0, r1))
                    if h1 and not r['mapping_identical']:
                        d1=new.evaluate(h1,speech[female],len(v));r1['speech']=metrics(v,d1,pairs,edges)
                    if h0 and not r['mapping_identical']:
                        d0=old.evaluate(h0,speech[female],len(v));r0['speech']=metrics(v,d0,pairs,edges)
                    r['expressions_identical']=bool(h0 and h1 and old.lib.expression_fingerprint(h0)==new.lib.expression_fingerprint(h1))
                    r['outside_repair_max_delta']=0.0
                    r['frame_penetration_worsening']=0.0
                    if h0 and h1 and not r['mapping_identical']:
                        r['frame_penetration_worsening']=penetration_worsening(v,d0,d1,pairs)
                        mask=np.zeros(len(v),np.uint8);new.lib.repair_mask(h1,mask)
                        r['outside_repair_max_delta']=float(np.max(np.abs(d1[:,mask==0]-d0[:,mask==0]))) if np.any(mask==0) else 0.0
                    if h0 and h1 and not r['mapping_identical']:
                        d0=old.evaluate(h0,speech[female],len(v),True)
                        d1=new.evaluate(h1,speech[female],len(v),True)
                        r['frame_penetration_worsening']=max(r['frame_penetration_worsening'],penetration_worsening(v,d0,d1,pairs))
                        r0['speech_limited']=metrics(v,d0,pairs,edges)
                        r1['speech_limited']=metrics(v,d1,pairs,edges)
                        delta=float(np.max(np.abs(d1[:,mask==0]-d0[:,mask==0]))) if np.any(mask==0) else 0.0
                        r['outside_repair_max_delta']=max(r['outside_repair_max_delta'],delta)
                    if args.strength_sweep and h0 and h1 and not r['mapping_identical']:
                        r['strength_sweep']=[]
                        for upper,lower in [(0,1),(2,1),(1,0),(1,2),(2,2)]:
                            row=dict(upper=upper,lower=lower,expression=10)
                            for label,mapper,head in [('old',old,h0),('new',new,h1)]:
                                output=np.empty((len(speech[female]),len(v),3),np.float32)
                                assert mapper.lib.evaluate_controls(head,len(speech[female]),speech[female].shape[1],speech[female],output,0,upper,lower,10)
                                row[label]=metrics(v,output,pairs,edges)
                            r['strength_sweep'].append(row)
                    r['status']='fitted' if h1 else 'native_fallback'
                    # A moved/deleted heuristic landmark is a measurement gap, not a pass.
                    if pairs:
                        ids=np.array([[p[0],p[1]] for p in pairs],int).flatten()
                        r['landmark_front_valid']=bool(np.min(v[ids,1])>np.quantile(v[:,1],.65))
                    else:r['landmark_front_valid']=False
                    results[digest]={k:r[k] for k in ['old','new','canonical_path','mapping_identical','status','landmark_front_valid','expressions_identical','outside_repair_max_delta','frame_penetration_worsening']}
                    if 'strength_sweep' in r:results[digest]['strength_sweep']=r['strength_sweep']
                if not npc:r['association']='unreferenced_asset'
                elif npc['deleted']:r['association']='deleted_NPC'
                elif npc['template_flags']&1:r['association']='template_traits_require_runtime_resolution'
                else:r['association']='direct_NPC'
            except Exception as e:r.update(status='blocked',error=str(e))
            finally:
                if h0:old.lib.release(h0)
                if h1:new.lib.release(h1)
            counts[r['status']]+=1;log.write(json.dumps(r,allow_nan=False)+'\n');log.flush()
            if index%100==0:print(index,len(tasks),dict(counts),'unique',len(results),'seconds',round(time.time()-start),flush=True)
    (ROOT/(args.out+'.tri-manifest.json')).write_text(json.dumps({k:v[3] for k,v in cache.items()},indent=2))
    print('DONE',dict(counts),'unique',len(results),'seconds',time.time()-start,flush=True)

if __name__=='__main__':main()
