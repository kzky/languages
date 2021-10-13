import time
from enum import Enum


class Node:


    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.childs = []
        self.parent = None


    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value})"


class NTree:


    def __init__(self, n, depth):
        self.root = None
        self.depth = depth
        self.n = n
        self.total = 0
        for d in range(self.depth):
            self.total += self.n ** d


    def construct(self, mode="BF"):
        if mode == "BF":
            self._breath_first_construct()
        elif mode == "DF":
            self._depth_first_construct()


    def _breath_first_construct(self):
        total_ = 1
        self.root = Node(0, 0)
        candidates = [self.root]
        while (self.total != total_):
            parent = candidates.pop(0)
            for i in range(self.n):
                child = Node(i, parent.depth + 1)
                child.parent = parent
                parent.childs.append(child)
                candidates.append(child)
                total_ += 1


    def _depth_first_construct(self):
        def _traverse(node, depth=0):
            if depth > self.depth:
                return
            for i in range(self.n):
                child = Node(i, depth)
                child.parent = node
                node.childs.append(child)
                _traverse(child, depth + 1)
                
        self.root = Node(0, 0)
        _traverse(self.root)
        

    def show(self, mode="BF"):
        if mode == "BF":
            self._breath_first_show()
        elif mode == "DF":
            self._depth_first_show()


    def _breath_first_show(self):
        total_ = 1
        candidates = [self.root]
        while (self.total != total_):
            parent = candidates.pop(0)
            for i in range(self.n):
                child = parent.childs[i]
                candidates.append(child)
                total_ += 1
                print(child)

                
    def _depth_first_show(self):
        def _traverse(node, depth=0):
            if depth > self.depth:
                return
            for i in range(self.n):
                child = node.childs[i]
                _traverse(child, depth + 1)
                print(child)
                
        _traverse(self.root)


def main():
    n = 3
    depth = 3
    mode = "DF"
    tree = NTree(n, depth)
    tree.construct(mode=mode)
    tree.show(mode=mode)

if __name__ == '__main__':
    main()
