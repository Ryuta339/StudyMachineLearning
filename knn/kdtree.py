"""
" KD Tree implementation
" 
" Reference:
    Wikipedia (https://ja.wikipedia.org/wiki/Kd%E6%9C%A8#:~:text=kd%E6%9C%A8%EF%BC%88%E8%8B%B1%3A%20kd%2D,%E5%88%86%E5%89%B2%E3%83%87%E3%83%BC%E3%82%BF%E6%A7%8B%E9%80%A0%E3%81%A7%E3%81%82%E3%82%8B%E3%80%82)
    scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)
    C言語によるkd-tree の実装 (https://qiita.com/fj-th/items/1bb2dc39f3088549ad6e)
    機械学習の2、kd-tree最近傍探索 (https://memo.soarcloud.com/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%EF%BC%92%E3%80%81kd-tree%E3%81%AB%E3%82%88%E3%82%8B%E6%9C%80%E8%BF%91%E5%82%8D%E6%8E%A2%E7%B4%A2/)
    Kd-treeを実装してみた (https://atkg.hatenablog.com/entry/2016/12/18/002353)

"""

import numpy as np
from matplotlib import pyplot as plt
from abc import ABC
import warnings

class Node:
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

    def __init__ (self, x, depth, is_leaf=False, left=None, right=None, parent=None):
        self.x = x
        self.depth = depth
        self.is_leaf = is_leaf
        self.left = left
        self.right = right
        self.parent = parent

    def print_node (self):
        if not self.is_leaf:
            self.left.print_node ()
        for i in range (self.depth):
            print ("+", end="")
        print (self.x)
        if not self.is_leaf and self.right!=None:
            self.right.print_node ()


class KDTree:
    """
    " This class represents KD-Tree
    " 
    " TODO:
        Create binary tree class as a super class
    "
    " Attribute:
        tree (Node) : The parent node of kd tree
    """

    def __init__  (self, data):
        self.tree = self.recursive_build (data, 0)

    def recursive_build (self, data, depth):
        n, dim = np.shape (data)

        if n<0:
            raise IndexError()

        if n==0:
            return None

        if n==1:
            # Leaf node
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

    def query (self, data, k=1):
       return _search_recursive (self.tree, data, k)

    # print
    def print_tree (self):
        self.tree.print_node ()

    # Visualize
    def plot_2d_tree (self, x_range, y_range):
        _plot_2d_tree_recursive (self.tree, x_range, y_range)


class Distance:
    @classmethod
    def distance (self, p1, p2):
        """
        Euclidian distance

        TODO:
            Create abstract distance class
            and create concrete (Euclidean or other) distance class
        """
        return np.sqrt(np.sum ((p1-p2)**2))

# Subroutine function
def _search_recursive (node, data, k=1, datalist=np.empty(0), dists=np.empty(0)):
    if node==None:
        return

    dim = node.x.size
    nodedata = node.x.reshape ((dim,))

    if data.size != dim:
        raise ValueError ("Dimension mismatch")
    data = data.reshape ((dim,))

    point = node.x
    dist = Distance.distance (point, data)
    
    # Update the k nearest neighbors
    # TODO:
    # This conditional branch is so ugly
    # Need to introduce a state pattern
    if datalist.size==0 or dists.size==0:
        datalist = nodedata.reshape((1,dim))
        dists = np.hstack ([dists, dist])
        
    elif len(datalist)==len(dists) and len(datalist) < k:
        dists = np.hstack ([dists, dist])
        datalist = np.vstack ([datalist, nodedata])

    elif len(datalist)==len(dists) and len(datalist) >= k:
        argmaxdist = np.argmax (dists)
        if dist < dists[argmaxdist]:
            dists[argmaxdist] = dist
            datalist[argmaxdist,:] = nodedata

    else:
        raise ValueError ("Dimension msimatch")

    # recursively search
    if node.is_leaf:
        return datalist, dists

    split_axis = node.depth % dim
    # 多次元の場合ここはバグのもとな気がする
    search_node = node.right if data[split_axis]>nodedata[split_axis] else node.left
    if search_node!=None:
        datalist, dists = _search_recursive (search_node, data, k, datalist, dists)

    if len(dists)<k or dist<=np.max(dists):
        other_node = node.left if data[split_axis]>nodedata[split_axis] else node.right
        if other_node!=None:
            datalist, dists = _search_recursive (other_node, data, k, datalist, dists)

    return datalist, dists


# Subroutine function to recursive
def _plot_2d_tree_recursive (node, x_range, y_range, ax=None):
    if node == None:
        return;

    dim, = node.x.shape
    if dim != 2:
        warnings.warn ("Dimension mismatch")
        return

    if len (x_range) != 2:
        warinigs.warn ("Dimension of x_range mismatch")
        return
    if len (y_range) != 2:
        warinigs.warn ("Dimension of y_range mismatch")
        return

    if ax == None:
        ax = plt.gca()

    split_axis = node.depth % dim
    cmap = plt.get_cmap ("tab10")

    line = np.ones ((2,)) * node.x[split_axis]
    if split_axis==0:
        ax.plot (line, y_range, color=cmap(node.depth%12))
        m_range = [x_range[0], node.x[split_axis], x_range[1]]
        _plot_2d_tree_recursive (node.left, m_range[:2], y_range, ax)
        _plot_2d_tree_recursive (node.right, m_range[1:], y_range, ax)
    else:
        ax.plot (x_range, line, color=cmap(node.depth%12))
        m_range = [y_range[0], node.x[split_axis], y_range[1]]
        _plot_2d_tree_recursive (node.left, x_range, m_range[:2], ax)
        _plot_2d_tree_recursive (node.right, x_range, m_range[1:], ax)

    plt.plot (node.x[0], node.x[1], 'xk')
    plt.xlim ([-1,1])
    plt.ylim ([-1,1])
    


# plot kd tree
if __name__ == '__main__':
    data = np.random.rand (10,2) * 2 - 1
    tree = KDTree (data)
    #tree.tree.print_node ()
    #tree.plot_2d_tree ([-1,1],[-1,1])
    plt.plot (data[:,0], data[:,1], 'kx')
    query = np.random.rand (2,)
    print (query)
    plt.plot (query[0], query[1,], 'or')
    datalist, dists = tree.query (query, k=3)
    for idx, p in enumerate (datalist):
        plt.plot ([query[0], p[0]], [query[1], p[1]], 'r--')
        # plt.plot (p[0], p[1], 'ok')

    thetas = np.linspace (0,2*3.14159,100)
    r = np.zeros ((2,100))
    r[0,:] = np.max(dists) * np.cos(thetas) + query[0]
    r[1,:] = np.max(dists) * np.sin(thetas) + query[1]
    plt.plot (r[0,:], r[1,:])
    plt.xlim ([-1.1,1.1])
    plt.ylim ([-1.1,1.1])
    plt.show()
