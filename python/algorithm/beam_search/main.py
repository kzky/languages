"""
This is a sample of beam search based on my guess and understanding for what
the beam search is.
"""

from beam_search import Node
from beam_search import BeamSearch
import numpy as np


def main():

    # T=5
    n_51 = Node(idx=51)
    n_52 = Node(idx=52)

    # T=4
    n_41 = Node(idx=41, next_nodes=[n_51])
    n_42 = Node(idx=42, next_nodes=[n_51])
    n_43 = Node(idx=43, next_nodes=[n_52])
    
    # T=3
    n_31 = Node(idx=31, next_nodes=[n_41, n_42, n_43])

    # T=2
    n_21 = Node(idx=21, next_nodes=[n_31])
    n_22 = Node(idx=22, next_nodes=[n_31])
    n_23 = Node(idx=23, next_nodes=[n_31])
    
    # T=1
    n_11 = Node(idx=11, next_nodes=[n_21, n_22])
    n_12 = Node(idx=12)
    n_13 = Node(idx=13, next_nodes=[n_23])
    
    # T=0
    n_root = Node(idx=0, next_nodes=[n_11, n_12, n_13])

    # Beam search
    n_beam = 3
    l_times = 5
    beam_search = BeamSearch(n_beam, l_times)
    seqs = beam_search.find(n_root)

    # Check nodes
    for i, seq in enumerate(seqs):
        print("Sequence {}".format(i))
        for n in seq:
            print(n.idx)
    
if __name__ == '__main__':
    main()


    
    
