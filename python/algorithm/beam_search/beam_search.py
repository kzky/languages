import copy

class Node(object):
    def __init__(self, idx, next_nodes=[]):
        """
        Parameters
        -----------------
        idx: int
           node index
        next_nodes: list of nodes

        """
        self.idx = idx
        self.next_nodes = next_nodes

class BeamSearch(object):

    def __init__(self, n_beam, l_times):
        """
        Parameters
        -----------------
        n_beam: int
          number of beams
        l_times: int
          length of time
        
        Attributes
        ----------------
        _n_beam: int
          number of beams
        _l_times: int
          length of time, excluding root node
        _l_times_1: int
          length of time, including root node
        _seqs: list of list
        _seq: list
        """

        self._n_beam = n_beam
        self._l_times = l_times
        self._l_times_1 = l_times + 1
        self._seqs = []
        self._seq = []

        self._used = False


    def find(self, node):
        """Find list of sequences

        Find list of sequences each of which is best sequences in terms of likelihood.

        Paramters
        ----------------
        node: Node
          Node is usually the root node.

        Return
        ----------
        seqs: list of list
           the result of search.
        """

        # Set as true because this method is destructive
        if self._used == True:
            raise Exception("You can not use this object again")

        self._used = True
        self._find(node)

        return self._seqs
        
    def _find(self, node):
        self._seq.append(node)
        for node in node.next_nodes:
            node = self._find(node)

        if (len(self._seq) != self._l_times_1):
            self._seq.pop()
            return

        if len(self._seq) == self._l_times_1:
            seq = copy.deepcopy(self._seq)
            self._seqs.append(seq)
            self._seq.pop()
            return
        
