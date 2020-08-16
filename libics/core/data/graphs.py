import collections


###############################################################################


class DirectedGraph(object):
    """
    A simple Python directed graph class.

    Stores the graph in a dictionary mapping vertex->set(vertex).
    """

    def __init__(self):
        self.__graph_dict = {}
        self.__reverse_graph = {}

    def vertices(self):
        """
        Returns the vertices of a graph.
        """
        return list(self.__graph_dict.keys())

    def add_vertex(self, vertex):
        """
        If the vertex `vertex` is not in `self.__graph_dict`,
        a key `vertex` with an empty list as a value is added
        to the dictionary. Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = set()
            self.__reverse_graph[vertex] = set()

    def add_edge(self, vertex_from, vertex_to):
        """
        Adds a directed edge from `vertex_from` to `vertex_to`.
        """
        for _v in (vertex_from, vertex_to):
            if _v not in self.__graph_dict:
                self.add_vertex(_v)
        self.__graph_dict[vertex_from].add(vertex_to)

    def get_edge_number(self, vertex):
        """
        Gets the number of edges connected at the given `vertex`.
        """
        return len(self.__graph_dict[vertex])

    def has_edge(self, vertex_from, vertex_to):
        return vertex_to in self.__graph_dict[vertex_from]

    def find_connected_vertices(self):
        """
        Finds all connected vertices.

        Returns
        -------
        connections : `list(list)`
            List of list containing vertices.
            Vertices within a deque are mutually connected.
        """
        visited = dict.fromkeys(self.__graph_dict, False)
        visited_rev = dict.fromkeys(self.__graph_dict, False)
        connections = []
        for vertex in self.vertices():
            if not visited[vertex]:
                _deque = collections.deque()
                self._iterate_graph_connections(
                    vertex, _deque, visited
                )
                self._iterate_graph_connections_rev(
                    vertex, _deque, visited, visited_rev
                )
                connections.append(list(_deque))
        return connections

    def _iterate_graph_connections(self, vertex, _deque, _visited):
        """
        Iterates through graph dictionary and appends all vertices connected
        to the given `vertex`.

        Note that `_deque` and `_visited` are modified upon iteration.
        Note that connections are appended to the right of the deque.

        Parameters
        ----------
        vertex : `hashable`
            Vertex key.
        _deque : `collections.deque`
            Double ended queue storing connection.
        _visited : `dict(hashable->bool)`
            Boolean dictionary storing whether vertex was visited.

        Returns
        -------
        _deque : `collections.deque`
            Double ended queue storing the connection.
        """
        if not _visited[vertex]:
            _visited[vertex] = True
            _deque.append(vertex)
            for _v in self.__graph_dict[vertex]:
                self._iterate_graph_connections(_v, _deque, _visited)
        return _deque

    def _iterate_graph_connections_rev(
        self, vertex, _deque, _visited, _visited_rev
    ):
        """
        Same as :py:meth:`_iterate_graph_connection`, but appends to the
        left of the deque and iterates in the reverse way.

        Parameters
        ----------
        _visited_rev : `dict(hashable->bool)`
            Boolean dictionary storing whether vertex was visited in
            the reverse iteration. The parameter `_visited` is used
            to check if the vertex needs to be added to the deque.
        """
        if not _visited_rev[vertex]:
            _visited_rev[vertex] = True
            if not _visited[vertex]:
                _deque.appendleft(vertex)
                _visited[vertex] = True
            for _v in self.__reverse_graph[vertex]:
                self._iterate_graph_connections_rev(
                    _v, _deque, _visited, _visited_rev
                )
        return _deque
