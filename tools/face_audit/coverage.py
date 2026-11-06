"""Account for every winning NPC record; never count absent assets as passing."""
import json,collections
from pathlib import Path
from assets import ROOT

def coverage(results):
    records=json.loads((ROOT/'records.json').read_text());npcs=records['npcs'];levels=records['leveled'];races=records['races']
    by={r['npc_id']:r for r in results if 'npc_id' in r};memo={}
    def resolve(ident,seen=frozenset()):
        if ident in seen:return frozenset(['CYCLE:'+str(ident)])
        if ident in memo:return memo[ident]
        seen=seen|{ident}
        if ident in levels:
            entries=levels[ident]['entries']
            out=frozenset().union(*(resolve(x,seen) for x in entries)) if entries else frozenset(['EMPTY_LEVEL:'+ident])
        elif ident in npcs:
            n=npcs[ident]
            out=resolve(n['template'],seen) if n['template_flags']&1 else frozenset([ident])
        else:out=frozenset(['MISSING:'+str(ident)])
        memo[ident]=out;return out
    rows=[]
    for ident,n in npcs.items():
        row={'id':ident,'edid':n['edid'],'source':n['source']}
        if n['deleted']:row['status']='deleted_record';rows.append(row);continue
        leaves=resolve(ident);items=[]
        for leaf in sorted(leaves):
            item={'id':leaf}
            if leaf not in npcs:item['status']='unresolved_template_or_list'
            else:
                actor=npcs[leaf];race=races.get(actor['race'],{});item.update(race=race.get('edid'),edid=actor['edid'])
                if race.get('facegen') is False:item['status']='non_facegen_race'
                elif actor.get('preset'):item['status']='chargen_preset_requires_runtime_generation'
                elif leaf not in by:item['status']='missing_baked_head'
                else:item['status']=by[leaf]['status'];item['asset']=by[leaf]['path']
            items.append(item)
        statuses={x['status'] for x in items}
        if statuses=={'non_facegen_race'}:row['status']='non_facegen_race'
        elif statuses <= {'non_facegen_race','fitted'}:row['status']='fitted_direct' if leaves=={ident} else 'fitted_template_choices'
        elif statuses <= {'non_facegen_race','fitted','native_fallback'}:row['status']='includes_native_fallback'
        elif statuses=={'chargen_preset_requires_runtime_generation'}:row['status']='chargen_preset_requires_runtime_generation'
        else:row['status']='coverage_gap'
        row['appearances']=items;rows.append(row)
    return rows

if __name__=='__main__':
    results=[json.loads(s) for s in (ROOT/'full.jsonl').read_text().splitlines()]
    rows=coverage(results);(ROOT/'npc-coverage.json').write_text(json.dumps(rows,indent=2))
    print(collections.Counter(r['status'] for r in rows))
