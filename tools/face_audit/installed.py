"""Inventory all installed MO2 source variants, including inactive/overridden mods.
These are hypothetical source-mod-over-active-profile combinations, not a load order.
"""
import os,json,struct
from pathlib import Path
from assets import ROOT,GAME,MO,PROFILE,records,key
from bsa import BSA
import esm

def main():
    roots=[GAME/'Data']+sorted(p for p in (MO/'mods').iterdir() if p.is_dir())+[MO/'overwrite']
    active={s[1:] for s in (PROFILE/'modlist.txt').read_text().splitlines() if s.startswith('+')}
    result={'scope':'All installed MO2 source variants; local morph assets preferred, active assets as fallback. Disabled combinations are hypothetical, not certified runtime winners. Downloads and unrelated disk locations are not enumerated.','roots':{},'tasks':[],'errors':[]}
    for ri,root in enumerate(roots):
        info={'active':root.name in active or root in [GAME/'Data',MO/'overwrite'],'parts':{},'tris':{}}
        for d,dirs,files in os.walk(root):
            if Path(d)==root:dirs[:]=[x for x in dirs if x.lower()=='meshes']
            for fn in files:
                p=Path(d)/fn;k=key(p.relative_to(root));suf=p.suffix.lower()
                src={'file':str(p)}
                if '\\facegeom\\' in k and suf=='.nif':result['tasks'].append({'path':k,'source':src,'root':str(root)})
                elif suf=='.tri':info['tris'].setdefault(k,[]).append(src)
                elif suf=='.bsa':
                    try:
                        a=BSA(p)
                        for ak in a.files:
                            ass={'archive':str(p),'path':ak}
                            if '\\facegeom\\' in ak and ak.endswith('.nif'):result['tasks'].append({'path':ak,'source':ass,'root':str(root)})
                            elif ak.endswith('.tri'):info['tris'].setdefault(ak,[]).append(ass)
                        a.f.close()
                    except Exception as e:result['errors'].append([str(p),str(e)])
                elif suf in ('.esm','.esp','.esl'):
                    try:
                        for ty,fid,flags,subs in records(p,{b'HDPT'}):
                            name=esm.cstr(esm.first(subs,'EDID') or b'');raw=esm.first(subs,'PNAM')
                            if not name or not raw or len(raw)!=4 or flags&32:continue
                            part={'edid':name,'type':struct.unpack('<I',raw)[0],'tri':None,'plugin':str(p)};mode=None
                            for t,v in subs:
                                if t=='NAM0':mode=struct.unpack('<I',v)[0]
                                elif t=='NAM1' and mode==1:part['tri']='meshes\\'+key(esm.cstr(v)).removeprefix('meshes\\')
                            info['parts'].setdefault(name.lower(),[]).append(part)
                    except Exception as e:result['errors'].append([str(p),str(e)])
        if info['parts'] or info['tris'] or any(t['root']==str(root) for t in result['tasks']):result['roots'][str(root)]=info
        if ri%100==0:print(ri,len(roots),'variants',len(result['tasks']),flush=True)
    (ROOT/'installed.json').write_text(json.dumps(result,indent=2));print('DONE',len(result['tasks']),'errors',result['errors'],flush=True)

if __name__=='__main__':main()
