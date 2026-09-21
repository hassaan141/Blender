import csv, json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[3]
OLD=ROOT/'docs/walk_ref/loop_runs'
sys.path.insert(0,str(OLD))
from selection_rules import old_selection, protocol_rejections
baseline=json.loads((OLD/'baseline/metrics.json').read_text())
rows=list(csv.DictReader((OLD/'leaderboard_pre_audit.csv').open()))
best='baseline'; best_m=baseline; audited=[]
for row in rows:
    name=row['run'];m=json.loads((OLD/name/'metrics.json').read_text())
    original=row['decision']
    if name=='baseline':reasons=[];decision='BASELINE'
    else:
        reasons=old_selection(m,best_m,baseline);decision='REJECT' if reasons else 'KEEP'
    row.update(original_decision=original,audited_against=best,decision=decision,reason='; '.join(reasons) or 'corrected fixed-baseline protocol passed',accel_regression_pct=(m['joint_accel_rms']/baseline['joint_accel_rms']-1)*100)
    audited.append(row)
    if decision=='KEEP':best=name;best_m=m
    if name!='baseline':
        p=OLD/name/'experiment.json';e=json.loads(p.read_text());e.setdefault('original_decision',e['decision']);e.setdefault('original_reason',e['reason']);e.update(decision=decision,reason=row['reason'],audit_comparison_checkpoint=row['audited_against']);p.write_text(json.dumps(e,indent=2))
for dest in [OLD/'leaderboard.csv',pathlib.Path(__file__).parent/'audit_existing.csv']:
    with dest.open('w') as f:
        w=csv.DictWriter(f,list(audited[0]),lineterminator="\n");w.writeheader();w.writerows(audited)
assert best=='run_06'
report={'best_valid_start':best,'checkpoint':str(OLD/best/'policy.pt'),'fixed_original_acceleration':baseline['joint_accel_rms'],'fixed_acceleration_ceiling':1.2*baseline['joint_accel_rms'],'valid_candidates':[r['run'] for r in audited[1:] if not protocol_rejections(json.loads((OLD/r['run']/'metrics.json').read_text()),baseline)],'all_speed_targets_met_by_any_valid_candidate':False,'provenance_note':'Run 06 was historically trained from now-invalid Run 05. That provenance is preserved, not rewritten. Its own measured checkpoint passes the fixed-baseline guards and improves on Run 01, the last valid retained checkpoint. This is a retrospective checkpoint audit, not a claim the historical training branches followed corrected selection.'}
(pathlib.Path(__file__).parent/'audit_existing.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
