import collections


###############################################################################


class FlaggedType:

    """
    Container for values with boolean flag.

    Provides a validity checker (range or subset based).
    Implements operators mathematical operators. On combinatory operation
    (e.g. `+`), flags are combined as `or` and conditions are dropped (`None`).

    Parameters
    ----------
    val
        Value to be stored.
    flag : bool
        Boolean flag.
    cond : tuple or list or None
        Value validity condition.
        tuple with length 2: range from min[0] to max[1].
        list: discrete set of allowed values.
        None: no validity check.
    """

    def __init__(self, val, flag=False, cond=None):
        self._val = None
        self._cond = None
        self.val = val
        self.flag = flag

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, v):
        if self._cond is not None:
            if type(self._cond) == tuple:
                if v < self._cond[0] or v > self._cond[1]:
                    raise ValueError("libics.util.types.FlaggedType.val: {:s}"
                                     .format(str(v)))
            else:
                if v not in self._cond:
                    raise ValueError("libics.util.types.FlaggedType.val: {:s}"
                                     .format(str(v)))
        self._val = v

    @property
    def cond(self):
        return self._cond

    @cond.setter
    def cond(self, c):
        if (c is None or type(c) == list or
                (type(c) == tuple and len(c) == 2 and c[0] <= c[1])):
            self._cond = c
        else:
            raise TypeError("libics.util.types.FlaggedType.cond: {:s}"
                            .format(str(c)))

    def invert(self):
        self.flag = not self.flag
        return self.flag

    def assign(self, other, diff_flag=False):
        """
        Copies the attributes of another `FlaggedType` object into the current
        object.

        Parameters
        ----------
        other : FlaggedType
            The object to be copied.
        diff_flag : bool
            Whether to copy the flag state or the differential flag
            state.
            `False`:
                Copies the flag state.
            `True`:
                Compares the values and sets the flag to
                `self.val != other.val`.
        """
        if diff_flag:
            self.flag = (self.val != other.val)
        else:
            self.flag = other.flag
        self.cond = other.cond
        self.val = other.val

    def set_val(self, val, diff_flag=True):
        """
        Sets the value without changing the condition.

        Parameters
        ----------
        val
            New value of flagged type.
        diff_flag : bool
            Whether to set a differential flag.
            `False`:
                Keeps the current flag state.
            `True`:
                Compares the values and sets the flag to
                `self.val != val`.
        """
        _old_val = self.val
        self.val = val
        if diff_flag:
            self.flag = (val != _old_val)

    def copy(self):
        """
        Returns an independent copy of itself.
        """
        return FlaggedType(self.val, flag=self.flag, cond=self.cond)

    def __eq__(self, other):
        return self.val == other.val

    def __ne__(self, other):
        return self.val != other.val

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __add__(self, other):
        return FlaggedType(
            self.val + other.val,
            flag=(self.flag or other.flag)
        )

    def __sub__(self, other):
        return FlaggedType(
            self.val - other.val,
            flag=(self.flag or other.flag)
        )

    def __mul__(self, other):
        return FlaggedType(
            self.val * other.val,
            flag=(self.flag or other.flag)
        )

    def __truediv__(self, other):
        return FlaggedType(
            self.val / other.val,
            flag=(self.flag or other.flag)
        )

    def __pow__(self, other):
        return FlaggedType(
            self.val**other.val,
            flag=(self.flag or other.flag)
        )

    def __neg__(self):
        return FlaggedType(-self.val, flag=self.flag, cond=self.cond)

    def __int__(self):
        return int(self.val)

    def __float__(self):
        return float(self.val)

    def __str__(self):
        return str(self.val)


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
