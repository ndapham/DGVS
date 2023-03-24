import pprint

from playground import playground_tubes
from aggregation.graph_buffering.dynamic_graph import RuanDynamicGraph

dynamic_graph = RuanDynamicGraph(q=3, h=1, p=10)

stress_test_tubes = playground_tubes
pprint.pprint(f'Num of tubes in testing: {len(stress_test_tubes)}')

dynamic_graph.run_pipeline(stress_test_tubes)
