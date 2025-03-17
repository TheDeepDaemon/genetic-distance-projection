import networkx as nx

def make_relations_graph(genome_data):
    graph = nx.DiGraph()

    for gid in genome_data:
        graph.add_node(gid)

    for gid, data in genome_data.items():
        for parent in data['parents']:
            graph.add_edge(int(parent), gid)

    return graph
