"""恢复实体链接并写入独立目录；旧产物不覆盖，所有阶段缺口明确保留。"""
import argparse,json,time
from pathlib import Path
from ese_agp.entities.recovery import restore_links
from ese_agp.entities.pipeline import atomic_json
from ese_agp.data.io import load_prepared, data_fingerprint
from ese_agp.graphs.behavior import graph_fingerprint, load_graph, save_graph
from ese_agp.graphs.semantic import build_semantic_graph
from ese_agp.graphs.fusion import fuse_graphs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--patches')
    parser.add_argument('--output',required=True)
    parser.add_argument('--graph-output',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--behavior',required=True)
    args=parser.parse_args();start=time.time()
    source=Path(args.source);out=Path(args.output);graphs=Path(args.graph_output)
    out.mkdir(parents=True,exist_ok=False);graphs.mkdir(parents=True,exist_ok=False)
    def read(name):return json.loads((source/(name+'.json')).read_text())
    old=read('entities');registry=read('entity_registry');dis=read('disambiguation')
    for artifact in (old,registry,dis):
        if artifact['fingerprint']!=graph_fingerprint(artifact):raise ValueError('Source fingerprint mismatch')
    docs=read('documents');chunks=read('joint_chunks')
    if docs['identity']!=old['identity'] or chunks['identity']!=old['identity']:raise ValueError('Source identity mismatch')
    if args.patches:
        patch=json.loads(Path(args.patches).read_text())
        if patch['source_identity']!=old['identity']:raise ValueError('Patch identity mismatch')
        chunks['results'].extend(patch['results'])
    failures=read('failures')['failures'];documents={d['item_id']:d['text'] for d in docs['documents']}
    nodes,relations,by_item=restore_links(chunks['results'],registry['entities'],dis['decisions'],failures,documents)
    data=load_prepared(args.data)
    if data_fingerprint(data)!=old['data_fingerprint']:raise ValueError('Dataset mismatch')
    artifact={**old,'scope':'pilot_or_partial','complete':False,'workflow_complete':False,
        'recovery_policy':'retain_grounded_links_entity_local_description_failure_v1',
        'source_entity_fingerprint':old['fingerprint'],
        'completed_products':len(by_item),'failed_products':len(documents)-len(by_item),
        'records':[{'item_id':i,'status':'success' if i in by_item else 'failed','workflow_status':'partial',
                    'entities':by_item.get(i,[])} for i in sorted(documents)]}
    artifact.pop('fingerprint',None)
    for name,body in [('entity_registry',{'entities':list(nodes.values())}),('relations',{'relations':list(relations.values())})]:
        body.update(data_fingerprint=old['data_fingerprint'],source_identity=old['identity'])
        body['fingerprint']=graph_fingerprint(body);atomic_json(out/(name+'.json'),body)
        if name=='entity_registry':artifact['registry_fingerprint']=body['fingerprint']
    artifact['fingerprint']=graph_fingerprint(artifact);atomic_json(out/'entities.json',artifact)
    semantic=build_semantic_graph(data,out/'entities.json',20,allow_partial=True,max_candidate_pairs=500000000)
    fused=fuse_graphs(load_graph(args.behavior,data),semantic,.5,allow_partial=True)
    save_graph(semantic,graphs/'semantic.json');save_graph(fused,graphs/'fused.json')
    report={'complete':False,'recovery_complete':True,'workflow_complete':False,
        'source':str(source),'output':str(out),'graph_output':str(graphs),
        'previous_success_records':old['completed_products'],
        'previous_linked_products':sum(r['status']=='success' and bool(r['entities']) for r in old['records']),
        'linked_products':len(by_item),
        'restored_products':len(set(by_item)-{r['item_id'] for r in old['records'] if r['status']=='success' and r['entities']}),
        'products_without_valid_entities':len(documents)-len(by_item),
        'historical_joint_extraction_failed_products':len({f['item_id'] for f in failures if f['stage']=='joint'}),
        'joint_extraction_failed_products':len({f['item_id'] for f in failures if f['stage']=='joint'}-set(by_item)),
        'empty_extraction_products':len(set(documents)-set(by_item)-{f['item_id'] for f in failures if f['stage']=='joint'}),
        'pending_descriptions':sum(e['description_status']=='pending' for e in nodes.values()),
        'pending_disambiguation_entities':sum(e['disambiguation_status']=='pending' for e in nodes.values()),
        'semantic_edges':semantic['num_edges'],'fused_edges':fused['num_edges'],
        'semantic_isolated':semantic['num_isolated'],'elapsed_seconds':time.time()-start,
        'llm_requests':0,'source_failures_preserved':len(failures)}
    atomic_json(out/'recovery_report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
