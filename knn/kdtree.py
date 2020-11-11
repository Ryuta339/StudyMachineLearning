"""
" KD Tree implementation
" 
" Reference:
    Wikipedia (https://ja.wikipedia.org/wiki/Kd%E6%9C%A8#:~:text=kd%E6%9C%A8%EF%BC%88%E8%8B%B1%3A%20kd%2D,%E5%88%86%E5%89%B2%E3%83%87%E3%83%BC%E3%82%BF%E6%A7%8B%E9%80%A0%E3%81%A7%E3%81%82%E3%82%8B%E3%80%82)
    scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)
    C言語によるkd-tree の実装 (https://qiita.com/fj-th/items/1bb2dc39f3088549ad6e)
    機械学習の2、kd-tree最近傍探索 (https://memo.soarcloud.com/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%EF%BC%92%E3%80%81kd-tree%E3%81%AB%E3%82%88%E3%82%8B%E6%9C%80%E8%BF%91%E5%82%8D%E6%8E%A2%E7%B4%A2/)

"""

import numpy as np
from matplotlib import pyplot as plt
import warnings

""" 
" Node class
" This class represents a node of a binary tree.
"
" Attributes:
    x       (float, ndarray): Data.
    depth   (int)           : Depth of this node. The root is 0.
    is_leaf (bool)          : is_leaf is true if this node is true.
    left    (Node, NoneType): Left node.
    right   (Node, NoneType): Right node.
    parent  (Node, NoneType): Parent node.
"""
class Node:
    def __init__ (self, x, depth, is_leaf=False, left=None, right=None, parent=None):
        self.x = x
        self.depth = depth
        self.is_leaf = is_leaf
        self.left = left
        self.right = right
        self.parent = parent

    def print_tree (self):
        if not self.is_leaf:
            self.left.print_tree ()
        for i in range (self.depth):
            print ("+", end="")
        print (self.x)
        if not self.is_leaf and self.right!=None:
            self.right.print_tree ()

"""
" This class represents KD-Tree
" 
" TODO:
    Create binary tree class as a super class
"
" Attribute:
    tree (Node) : The parent node of kd tree
"""
class KDTree:
    def __init__  (self, data):
        self.tree = self.recursive_build (data, 0)

    def recursive_build (self, data, depth):
        n, dim = np.shape (data)

        if n<0:
            raise IndexError()

        if n==0:
            return None

        if n==1:
            return Node (data[0], depth, True)

        split_axis = depth % dim
        sorted_data = data[np.argsort(data,axis=0)[:,split_axis]]
        median = n // 2
        left = self.recursive_build (sorted_data[:median], depth+1)
        right = self.recursive_build (sorted_data[median+1:], depth+1)
        node = Node (sorted_data[median], depth, False, left, right)

        # back pointer
        left.parent = node
        if right!=None:
            right.parent = node

        return node

    # Visualize
    def plot_2d_tree (self, xlow, xhigh, ylow, yhigh):
        _plot_2d_tree_recursive (self.tree, xlow, xhigh, ylow, yhigh)

# Subroutine method to recursive
def _plot_2d_tree_recursive (node, xlow, xhigh, ylow, yhigh, ax=None):
    if node == None:
        return;

    dim, = node.x.shape
    if dim != 2:
        warnings.warn ("Dimension mismatch")
        return

    if ax == None:
        ax = plt.gca()

    split_axis = node.depth % dim
    cmap = plt.get_cmap ("tab10")

    line = np.ones ((2,)) * node.x[split_axis]
    if split_axis==0:
        ax.plot (line, [ylow, yhigh], color=cmap(node.depth%12))
        _plot_2d_tree_recursive (node.left, xlow, node.x[split_axis], ylow, yhigh, ax)
        _plot_2d_tree_recursive (node.right, node.x[split_axis], xhigh, ylow, yhigh, ax)
    else:
        ax.plot ([xlow, xhigh], line, color=cmap(node.depth%12))
        _plot_2d_tree_recursive (node.left, xlow, xhigh, ylow, node.x[split_axis], ax)
        _plot_2d_tree_recursive (node.right, xlow, xhigh, node.x[split_axis], yhigh, ax)

    plt.plot (node.x[0], node.x[1], 'xk')
    plt.xlim ([-1,1])
    plt.ylim ([-1,1])
    


# plot kd tree
if __name__ == '__main__':
    data = np.random.rand (8,2) * 2 - 1
    tree = KDTree (data)
    tree.tree.print_tree ()
    tree.plot_2d_tree (-1,1,-1,1)
    plt.show()
