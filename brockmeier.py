import numpy as np
import networkx as nx

from correl_info.info import bures_info


def make_brockmeier_toy_graphs():
    """
    Make graphs for figures 9 and 10 of Brockmeier et al. 2017.
    These are a good check for whether the Bures informativeness
     implementation is working.
    """

    # non-isomorphic graphs with 5 vertices
    g1 = nx.Graph()
    g1.add_nodes_from(range(5))
    g1.add_edge(1, 2)
    g1.add_edge(3, 4)

    g2 = nx.Graph()
    g2.add_nodes_from(range(5))
    g2.add_edge(0, 1)
    g2.add_edge(1, 2)
    g2.add_edge(3, 4)

    g3 = nx.Graph()
    g3.add_nodes_from(range(5))
    g3.add_edge(0, 1)
    g3.add_edge(1, 2)
    g3.add_edge(2, 0)
    g3.add_edge(3, 4)

    # non-isomorphic graphs with 6 vertices

    g4 = nx.Graph()
    g4.add_nodes_from(range(6))
    g4.add_edge(0, 1)
    g4.add_edge(1, 2)
    g4.add_edge(2, 3)
    g4.add_edge(3, 4)
    g4.add_edge(4, 5)

    g5 = nx.Graph()
    g5.add_nodes_from(range(6))
    g5.add_edge(0, 1)
    g5.add_edge(1, 2)
    g5.add_edge(2, 3)
    g5.add_edge(1, 4)
    g5.add_edge(4, 5)

    g6 = nx.Graph()
    g6.add_nodes_from(range(6))
    g6.add_edge(0, 2)
    g6.add_edge(1, 2)
    g6.add_edge(2, 3)
    g6.add_edge(3, 4)
    g6.add_edge(4, 5)

    g7 = nx.Graph()
    g7.add_nodes_from(range(6))
    g7.add_edge(0, 2)
    g7.add_edge(1, 2)
    g7.add_edge(2, 3)
    g7.add_edge(3, 4)
    g7.add_edge(3, 5)

    g8 = nx.Graph()
    g8.add_nodes_from(range(6))
    g8.add_edge(0, 3)
    g8.add_edge(1, 3)
    g8.add_edge(2, 3)
    g8.add_edge(3, 4)
    g8.add_edge(4, 5)

    g9 = nx.Graph()
    g9.add_nodes_from(range(6))
    g9.add_edge(0, 1)
    g9.add_edge(1, 2)
    g9.add_edge(1, 3)
    g9.add_edge(1, 4)
    g9.add_edge(1, 5)

    graphs = {
        'fig9': [g1, g2, g3],
        'fig10': [g4, g5, g6, g7, g8, g9],
    }

    return graphs


def compute_brockmeier_toy_graphs_info_diffs():
    graphs = make_brockmeier_toy_graphs()
    brockmeier_infos = {
        'fig9': [0.143, 0.144, 0.136],
        'fig10': [0.056, 0.047, 0.044, 0.034, 0.029, 0.007],
    }
    fig_diffs = {}
    for fig_n in [9, 10]:
        fig_key = f'fig{fig_n}'
        fig_graphs = graphs[fig_key]
        binfos = brockmeier_infos[fig_key]
        my_infos = []
        for g in fig_graphs:
            lap = np.array(nx.normalized_laplacian_matrix(g).todense())
            np.fill_diagonal(lap, 1)
            my_infos.append(np.round(bures_info(lap), 3))
        diffs = np.array(binfos) - np.array(my_infos)
        fig_diffs[fig_key] = diffs
    return fig_diffs
