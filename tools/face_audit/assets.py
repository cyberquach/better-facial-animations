"""Local, read-only MO2 asset/record census. All outputs stay beside this tool."""
from pathlib import Path
import sys, os, struct, zlib, json, hashlib
sys.path.insert(0,str(Path(__file__).parent/'lib'))
from bsa import BSA
import esm, nif_uv

ROOT=Path(os.environ["BFA_AUDIT_OUTPUT"]).resolve()
GAME=Path(os.environ['BFA_AUDIT_GAME']).resolve()
MO=GAME/'MO2'
if os.environ.get('BFA_AUDIT_PROFILE'):
    PROFILE=MO/'profiles'/os.environ['BFA_AUDIT_PROFILE']
else:
    setting=next(s for s in (MO/'ModOrganizer.ini').read_text().splitlines() if s.startswith('selected_profile='))
    name=setting.split('=',1)[1]
    if name.startswith('@ByteArray(') and name.endswith(')'):name=name[11:-1]
    PROFILE=MO/'profiles'/name
def key(p): return str(p).replace('/','\\').lower()
def sha(b): return hashlib.sha256(b).hexdigest()

def records(path,want=None):
    b=path.read_bytes()
    def walk(i,end):
        while i<end:
            typ,size=struct.unpack_from('<4sI',b,i)
            if typ==b'GRUP':
                yield from walk(i+24,i+size);i+=size;continue
            flags,fid=struct.unpack_from('<II',b,i+8)
            if typ in (want or (b'TES4',b'NPC_',b'HDPT',b'RACE',b'LVLN')):
                data=b[i+24:i+24+size]
                if flags&0x40000:data=zlib.decompress(data[4:])
                yield typ.decode(),fid,flags,list(esm.subrecords(data))
            i+=24+size
    yield from walk(0,len(b))

class Assets:
    def __init__(self, cached=False):
        mods=[s[1:] for s in (PROFILE/'modlist.txt').read_text().splitlines() if s.startswith('+')]
        self.roots=[GAME/'Data']+[MO/'mods'/n for n in reversed(mods)]+[MO/'overwrite']
        self.files={}; self.archives={}; self.errors=[]; self.archive_assets={}
        for root in self.roots:
            for d,dirs,files in os.walk(root):
                rel=Path(d).relative_to(root)
                if rel==Path('.'):
                    dirs[:]=[x for x in dirs if x.lower()=='meshes']
                for n in files:
                    p=Path(d)/n; k=key(p.relative_to(root))
                    if p.suffix.lower() in ('.esm','.esp','.esl','.bsa','.nif','.tri','.ini'):
                        self.files[k]=p
        lines=(PROFILE/'plugins.txt').read_text().splitlines()
        on={s[1:].lower() for s in lines if s.startswith('*')}
        base={'skyrim.esm','update.esm','dawnguard.esm','hearthfires.esm','dragonborn.esm'}
        # Official creation content is auto-loaded by Skyrim.ccc rather than stars.
        ccc=GAME/'Skyrim.ccc'
        if ccc.exists():base.update(x.strip().lower() for x in ccc.read_text().splitlines() if x.strip())
        self.plugins=[s.strip().lower() for s in (PROFILE/'loadorder.txt').read_text().splitlines() if s.strip() and not s.startswith('#') and s.strip().lower() in on|base]
        # archives.txt is an enabled archive membership list, not plugin load order.
        listed=[s.strip().lower() for s in (PROFILE/'archives.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
        order={Path(p).stem:i for i,p in enumerate(self.plugins)}
        def rank(n):
            stem=Path(n).stem
            matches=[(i,k) for k,i in order.items() if stem==k or stem.startswith(k+' - ')]
            if matches:return max(matches)[0],n
            return -1,n
        # MO2 leaves archives.txt empty when the game manages archives. Include
        # INI resources and archives associated with enabled plugins explicitly.
        ini_archives=[]
        for ini in [PROFILE/'skyrim.ini', PROFILE/'initweaks.ini']:
            if ini.exists():
                for line in ini.read_text(errors='replace').splitlines():
                    if line.lower().startswith('sresourcearchivelist') and '=' in line:
                        ini_archives += [v.strip().lower() for v in line.split('=',1)[1].split(',') if v.strip()]
        associated=[n for n in self.files if n.endswith('.bsa') and rank(n)[0]>=0]
        self.archive_order=list(dict.fromkeys(ini_archives+sorted(set(listed+associated),key=rank)))
        self.unassociated_archives=[]
        for n in self.archive_order:
            if n not in self.files:self.errors.append(['missing_archive',n]);continue
            try:
                a=BSA(self.files[n]);self.archives[n]=a
                if rank(n)[0]<0 and not n.startswith('skyrim - '):self.unassociated_archives.append(n)
                for k in a.files:
                    if k.endswith(('.nif','.tri')):self.archive_assets.setdefault(k,[]).append(n)
            except Exception as e:self.errors.append(['archive',n,str(e)])
        self.npcs={};self.parts={};self.races={};self.leveled={};self.part_names={};self.plugin_hashes={}
        if cached:
            saved=json.loads((ROOT/'records.json').read_text())
            assert saved['plugins']==self.plugins
            for n,digest in saved['plugin_hashes'].items():
                assert sha(self.files[n].read_bytes())==digest, 'plugin changed: '+n
            for name in ('npcs','parts','races','plugin_hashes','leveled'):setattr(self,name,saved.get(name,{}))
            for p in self.parts.values():
                if not p['deleted']:self.part_names.setdefault(p['edid'].lower(),[]).append(p)
            return
        for pi,n in enumerate(self.plugins):
            p=self.files.get(n)
            if not p:self.errors.append(['missing_plugin',n]);continue
            masters=[]
            def form(fid):
                idx=fid>>24
                if idx>len(masters):return 'UNRESOLVED:%08x'%fid
                return (masters+[n])[idx]+':%06x'%(fid&0xffffff)
            try:
                self.plugin_hashes[n]=sha(p.read_bytes())
                for ty,fid,flags,subs in records(p):
                    if ty=='TES4':masters=[esm.cstr(v).lower() for t,v in subs if t=='MAST'];continue
                    ident=form(fid);r={'id':ident,'source':n,'deleted':bool(flags&32),'edid':esm.cstr(esm.first(subs,'EDID') or b'')}
                    if ty=='NPC_':
                        ac=esm.first(subs,'ACBS') or bytes(24)
                        r.update(female=bool(struct.unpack_from('<I',ac)[0]&1),preset=bool(struct.unpack_from('<I',ac)[0]&4),template_flags=struct.unpack_from('<H',ac,18)[0] if len(ac)>=20 else 0)
                        r['parts']=[form(struct.unpack('<I',v)[0]) for t,v in subs if t=='PNAM' and len(v)==4]
                        for t,field in [('RNAM','race'),('TPLT','template')]:
                            v=esm.first(subs,t);r[field]=form(struct.unpack('<I',v)[0]) if v and len(v)==4 else None
                        full=esm.first(subs,'FULL');r['name_raw']=full.hex() if full else ''
                        self.npcs[ident]=r
                    elif ty=='HDPT':
                        v=esm.first(subs,'PNAM');r['type']=struct.unpack('<I',v)[0] if v and len(v)==4 else None
                        r['model']=esm.cstr(esm.first(subs,'MODL') or b'');r['tri']=None
                        mode=None
                        for t,v in subs:
                            if t=='NAM0':mode=struct.unpack('<I',v)[0]
                            if t=='NAM1' and mode==1:r['tri']='meshes\\'+key(esm.cstr(v)).removeprefix('meshes\\')
                        self.parts[ident]=r
                    elif ty=='LVLN':
                        r['entries']=[form(struct.unpack_from('<I',v,4)[0]) for t,v in subs if t=='LVLO' and len(v)>=8]
                        self.leveled[ident]=r
                    else:
                        raw=esm.first(subs,'DATA')
                        r['flags']=struct.unpack_from('<I',raw,32)[0] if raw and len(raw)>=36 else None
                        r['facegen']=bool(r['flags']&2) if r['flags'] is not None else None
                        self.races[ident]=r
            except Exception as e:self.errors.append(['plugin_parse',n,str(e)])
            if pi%100==0:print('plugins',pi,len(self.plugins),flush=True)
        for p in self.parts.values():
            if not p['deleted']:self.part_names.setdefault(p['edid'].lower(),[]).append(p)

    def read(self,k):
        k=key(k)
        if k in self.files:
            p=self.files[k];return p.read_bytes(),{'path':k,'loose':str(p)}
        names=self.archive_assets.get(k,[])
        if not names:raise FileNotFoundError(k)
        n=names[-1]
        return self.archives[n].read(k),{'path':k,'archive':str(self.files[n]),'candidates':names,'unassociated':n in self.unassociated_archives}

    def head_paths(self):
        return sorted(k for k in self.files.keys()|self.archive_assets.keys() if '\\facegeom\\' in k and k.endswith('.nif'))

def named_shapes(d):
    # String table immediately follows block sizes; preserve exact per-shape EDID.
    o=d.index(b'\n')+1;_,_,_,nb=struct.unpack_from('<IBII',d,o);o+=13
    bs,=struct.unpack_from('<I',d,o);o+=4
    for _ in range(3+(bs>=103)):o+=1+d[o]
    nt,=struct.unpack_from('<H',d,o);o+=2
    for _ in range(nt):n,=struct.unpack_from('<I',d,o);o+=4+n
    o+=nb*6;ns,_=struct.unpack_from('<II',d,o);o+=8;strings=[]
    for _ in range(ns):n,=struct.unpack_from('<I',d,o);strings.append(d[o+4:o+4+n].decode('utf8','replace'));o+=4+n
    _,types,sizes,o=nif_uv._header(d);names=[]
    for ty,sz in zip(types,sizes):
        if ty in ('BSTriShape','BSDynamicTriShape','BSSubIndexTriShape'):
            i,=struct.unpack_from('<I',d,o);names.append(strings[i] if i<len(strings) else '')
        o+=sz
    shapes=nif_uv.shapes(d)
    assert len(names)==len(shapes)
    for s,n in zip(shapes,names):s['name']=n
    return shapes

if __name__=='__main__':
    a=Assets()
    out={'plugins':a.plugins,'plugin_hashes':a.plugin_hashes,'archive_order':a.archive_order,'unassociated_archives':a.unassociated_archives,'errors':a.errors,'npcs':a.npcs,'parts':a.parts,'races':a.races,'leveled':a.leveled}
    (ROOT/'records.json').write_text(json.dumps(out,indent=2))
    print({k:len(v) for k,v in out.items()},flush=True)
