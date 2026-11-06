"""Fail closed on lost fits, worsening measured aperture or missing coverage."""
import collections,json,os,sys
from pathlib import Path
from coverage import coverage
from checks import surface_regressions
out=Path(os.environ['BFA_AUDIT_OUTPUT']);summary={}
for scope in ['active','installed']:
 for model in [2,3]:
  files=sorted(out.glob(f'{scope}-v{model}-*.jsonl'))
  if not files:continue
  rows=[json.loads(line) for p in files for line in p.read_text().splitlines()]
  flags=[]
  for r in rows:
   a,b=r.get('old',{}),r.get('new',{})
   reasons=[]
   if a.get('accepted') and not b.get('accepted'):reasons.append('lost_fit')
   if a.get('accepted') and b.get('accepted'):
    if not r.get('expressions_identical'):reasons.append('authored_expression_changed')
    if r.get('outside_repair_max_delta',0)!=0:reasons.append('unrelated_vertex_changed')
    if r.get('frame_penetration_worsening',0)>1e-6:reasons.append('frame_projected_penetration_worse')
   if a.get('accepted') and b.get('accepted') and not r.get('mapping_identical'):
    for mode in ['speech','speech_limited']:
     sa,sb=a.get(mode,{}),b.get(mode,{})
     if not sb.get('finite'):reasons.append('nonfinite_or_unmeasured')
     surface_a,surface_b=sa.get('upper_surface',{}),sb.get('upper_surface',{})
     reasons.extend(mode+'_'+reason for reason in surface_regressions(surface_a,surface_b))
     if any(y<x-1e-6 for x,y in zip(sa.get('pair_peak_increase',[]),sb.get('pair_peak_increase',[]))):reasons.append('individual_opening_pair_reduced')
     if sa.get('opening_ratio',0)>0 and sb.get('opening_ratio',0)<sa['opening_ratio']-1e-6:reasons.append('opening_reduced')
     if sb.get('minimum_gap',0)<min(0,sa.get('minimum_gap',0))-1e-6:reasons.append('projected_penetration_worse')
    if not r.get('landmark_front_valid'):reasons.append('unreviewed_aperture_landmarks')
   for sweep in r.get('strength_sweep', []):
    label='strength_'+str(sweep['upper'])+'_'+str(sweep['lower'])
    before,after=sweep['old'],sweep['new']
    if not after.get('finite'):reasons.append(label+'_nonfinite')
    reasons.extend(label+'_'+reason for reason in surface_regressions(before.get('upper_surface',{}),after.get('upper_surface',{})))
   if reasons:flags.append({'path':r['path'],'source':r.get('variant_root'),'reasons':reasons,'old':a,'new':b})
  key=f'{scope}-v{model}'
  summary[key]={'assets':len(rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),
    'unique_inputs':len({r.get('input_sha256') for r in rows if r.get('input_sha256')}),
    'identical':sum(bool(r.get('mapping_identical')) for r in rows),'changed':sum(bool(r.get('old',{}).get('accepted') and r.get('new',{}).get('accepted') and not r.get('mapping_identical')) for r in rows),'flagged':len(flags)}
  (out/(key+'-flags.json')).write_text(json.dumps(flags,indent=2))
  if scope=='active':
   c=coverage(rows);(out/(key+'-npc-coverage.json')).write_text(json.dumps(c,indent=2));summary[key]['npc_coverage']=dict(collections.Counter(r['status'] for r in c))
summary['status']='regression_review_required' if any(v.get('flagged') for v in summary.values()) else 'blocked_coverage' if any(v.get('statuses',{}).get('blocked') or v.get('npc_coverage',{}).get('coverage_gap') or v.get('npc_coverage',{}).get('chargen_preset_requires_runtime_generation') for v in summary.values()) else 'pass'
summary['limitations']='Skin-only geometry; aperture pairs are independent TRI heuristics, not anatomical ground truth. Fitted is not a pass for in-game appearance. Missing assets, runtime-generated heads, other shapes and untested expressions remain coverage gaps.'
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

sys.exit({'pass':0,'regression_review_required':1,'blocked_coverage':2}[summary['status']])
