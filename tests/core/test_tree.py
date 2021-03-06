import pytest
import networkx
import numpy

from fragile.core.tree import NetworkxTree, HistoryTree


def random_powerlaw():
    g = networkx.DiGraph()
    t = networkx.random_powerlaw_tree(500, gamma=3, tries=1000, seed=160290)
    return networkx.compose(g, t)


def small_tree():
    node_data = {"a": numpy.arange(10), "b": numpy.zeros(10)}
    edge_data = {"c": numpy.ones(10)}
    g = networkx.DiGraph()
    for i in range(8):
        g.add_node(i, **node_data)

    g.add_edge(0, 1, **edge_data)
    g.add_edge(1, 2, **edge_data)
    g.add_edge(2, 3, **edge_data)
    g.add_edge(2, 4, **edge_data)
    g.add_edge(2, 5, **edge_data)
    g.add_edge(3, 6, **edge_data)
    g.add_edge(3, 7, **edge_data)
    return g


@pytest.fixture(params=[random_powerlaw, small_tree], scope="function")
def tree(request):
    tree = HistoryTree()
    tree.data = request.param()
    return tree


class TestNetworkxTree:
    def test_init(self, tree):
        pass

    def test_reset_graph(self, tree):
        node_data = {"miau": 2104}
        tree.reset_graph(root_id=421, node_data=node_data, epoch=0)
        assert tree.root_id == 421
        assert isinstance(tree.data.nodes[421], dict)
        assert tree.data.nodes[421]["epoch"] == 0
        assert len(tree) == 1
        assert tree.data.nodes[421]["miau"] == 2104

    def test_append_leaf(self, tree):
        node_data = {"node": numpy.arange(10)}
        edge_data = {"edge": False}
        leaf_id = -421
        epoch = 123
        tree.append_leaf(
            leaf_id=leaf_id,
            parent_id=tree.root_id,
            node_data=node_data,
            edge_data=edge_data,
            epoch=epoch,
        )
        assert (tree.data.nodes[leaf_id]["node"] == node_data["node"]).all()
        assert tree.data.nodes[leaf_id]["epoch"] == epoch
        assert tree.data.edges[(tree.root_id, leaf_id)] == edge_data
        assert leaf_id in tree.leafs
        assert tree.root_id not in tree.leafs

    def test_prune_dead_branches(self, tree):
        leafs = tree.get_leaf_nodes()
        tree.prune_dead_branches(dead_leafs=set(leafs[:2]), alive_leafs=set(leafs[2:]))
        for le in leafs[:2]:
            assert le not in tree.leafs
            assert le not in tree.data

        assert networkx.is_tree(tree.data)

    def test_get_parent(self, tree):
        node = tuple(tree.data.nodes)[1]
        child = list(tree.data.out_edges(node))[0][1]
        assert node == tree.get_parent(child)

    def test_compose_not_crashes(self, tree):
        other = NetworkxTree()
        other.data = small_tree()
        tree.compose(other)

    def test_get_path_node_ids(self, tree):
        leafs = tree.get_leaf_nodes()
        path = tree.get_path_node_ids(leafs[0])
        other_path = tree.get_branch(leafs[0])
        assert path == other_path

    def test_path_data_generator(self, tree):
        leaf = tree.get_leaf_nodes()[0]
        path = tree.get_path_node_ids(leaf)
        data = list(tree.path_data_generator(path))
        assert len(data) == len(path) - 1
        assert len(data[0]) == 2
        for x in data:
            assert isinstance(x, tuple)
            for y in x:
                assert isinstance(y, dict)

        data = list(tree.path_data_generator(path, return_children=True))
        assert len(data) == len(path) - 1
        assert len(data[0]) == 3
        for x in data:
            assert isinstance(x, tuple)
            for y in x:
                assert isinstance(y, dict)

    def test_random_nodes_generator(self, tree):
        data = list(tree.random_nodes_generator())
        assert len(data) == len(tree.data.nodes) - 1
        assert len(data[0]) == 2
        for x in data:
            assert isinstance(x, tuple)
            for y in x:
                assert isinstance(y, dict)

        data = list(tree.random_nodes_generator(return_children=True))
        assert len(data) == len(tree.data.nodes) - 1
        assert len(data[0]) == 3
        for x in data:
            assert isinstance(x, tuple)
            for y in x:
                assert isinstance(y, dict)
