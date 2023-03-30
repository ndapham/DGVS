from aggregation.graph_buffering.dynamic_graph import RuanDynamicGraph
from aggregation.graph_building.relations import RuanRelationsMap
from utils.io import create_json_file, load_tubes_from_json_file

meta_txt_path = "/home/pducanh/Desktop/pcgvs/data/meta.txt"
frame_json_path = "/home/pducanh/Desktop/pcgvs/data/meta.json"

frames_dict = create_json_file(meta_txt_path, frame_json_path)
tubes = load_tubes_from_json_file(frame_json_path)

print("Calculating the relations map")
relations = RuanRelationsMap(tubes=tubes[:10])
relation_dict = relations.save_as_json_dict("/home/pducanh/Desktop/pcgvs/data/ruan_relations.json")
dynamic_graph = RuanDynamicGraph(q=3, h=1, p=2)
dynamic_graph.run_pipeline(tubes[:10])
for tube in dynamic_graph.output_tubes:
    print(f'tube_id:{tube.tag} - color:{tube.color}')