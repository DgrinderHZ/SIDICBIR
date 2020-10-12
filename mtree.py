import abc
from heapq import heappush, heappop
import collections
from itertools import combinations, islice


#_____________________________________ Algorithmes de promotion et de partition ___________________________________#
def M_LB_DIST_confirmed(entries, current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Confirmed.
    Return the object that is the furthest apart from current_routing_entry
    using only precomputed distances stored in the entries.

    This algorithm does not work if current_routing_entry is None and the
    distance_to_parent in entries are None. This will happen when handling
    the root. In this case the work is delegated to M_LB_DIST_non_confirmed.

    arguments:
        entries N: set of entries from which two routing objects must be promoted.

        current_routing_entry: the routing_entry that was used
        for the node containing the entries previously.
        None if the node from which the entries come from is the root.

        d: distance function.
    """
    if current_routing_entry is None or any(e.distance_to_parent is None for e in entries):
        return M_LB_DIST_non_confirmed(entries,current_routing_entry,d)

    # if entries contain only one element or two elements twice the same, then
    # the two routing elements returned could be the same. (could that happen?)
    new_entry = max(entries, key=lambda e: e.distance_to_parent)
    return (current_routing_entry.obj, new_entry.obj)


def M_LB_DIST_non_confirmed(entries, unused_current_routing_entry, d):
    """Promotion algorithm. Maximum Lower Bound on DISTance. Non confirmed.
    Compares all pair of objects (in entries) and select the two who are
    the furthest apart.
    """
    objs = map(lambda e: e.obj, entries)
    return max(combinations(objs, 2), key=lambda two_objs: d(*two_objs))


#If the routing objects are not in entries it is possible that
#all the elements are in one set and the other set is empty.
def generalized_hyperplane(entries, routing_object1, routing_object2, d):
    """Partition algorithm.
    Each entry is assigned to the routing_object to which it is the closest.
    This is an unbalanced partition strategy.
    Return a tuple of two elements. The first one is the set of entries
    assigned to the routing_object1 while the second is the set of entries
    assigned to the routing_object2"""
    partition = (set(), set())
    for entry in entries:
        partition[d(entry.obj, routing_object1) > d(entry.obj, routing_object2)].add(entry)

    if not partition[0] or not partition[1]:
        # TODO
        partition = (set(islice(entries, len(entries)//2)),set(islice(entries, len(entries)//2, len(entries))))

    return partition


#_____________________________________ Classe M-Tree ___________________________________#
class MTree(object):
    """
    MTree(d, max_nodes): Class principal pour instancier une structure de donnees M-Tree

    d: distance utilisee
    max_nodes: M
    ____________________________________
    ____________________________________
    """
    def __init__(self, d, max_nodes=4, promote=M_LB_DIST_confirmed, partition=generalized_hyperplane):
        """
        Create a new MTree.
        Arguments:
        d: distance function.
        max_node_size: M optional. Maximum number of entries in a node of
            the M-tree
        promote: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines given the set of entries which two entries should be
            used as routing object to represent the two nodes in the
            parent node.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        partition: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines to which of the two routing object each entry of the
            split node should go.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        """
        if not callable(d):
            #Why the hell did I put this?
            #This is python, we use dynamic typing and assumes the user
            #of the API is smart enough to pass the right parameters.
            raise TypeError('d is not a function')
        if max_nodes < 2:
            raise ValueError('max_node_size must be >= 2 but is %d' % max_nodes)
        self.d = d
        self.max_nodes = max_nodes
        self.promote = promote
        self.partition = partition
        self.size = 0
        self.root = LeafNode(self)


    def __len__(self):
        """
            Return taille de l'arbre
        """
        return self.size

    def add(self, obj):
        self.root.add(obj)
        self.size += 1

    def add_all(self, iterable):
        for obj in iterable:
            self.add(obj)

    def k_NN_search(self, query_obj, k=1):
        """
        Methode k_NN_search:
            Algorithme de recherche K-plus proches voisins
        Args:
            query_obj: Q
            k: nombre de voisins

        Return the k objects the most similar to query_obj.
        Implementation of the k-Nearest Neighbor algorithm.
        Returns a list of the k closest elements to query_obj, ordered by
        distance to query_obj (from closest to furthest).
        If the tree has less objects than k, it will return all the
        elements of the tree.
        """
        k = min(k, len(self)) # on prend k < la taille de M-tree
        if k == 0: return []
        pr = []
        # Initialiser pr avec le Noeud root et dmin et dquery a 0
        heappush(pr, PrEntry(self.root, 0, 0))
        # Un objet qui va contenir les k plus proches voisins
        nn = NN(k)
        while pr:
            # NextNode = ChooseNode(PR);
            prEntry = heappop(pr)
            if(prEntry.dmin > nn.search_radius()):
                # best candidate is too far, we won't have better a answer
                # we can stop
                break
            # k_NN_NodeSearch(NextNode,Q,k);
            prEntry.tree.search(query_obj, pr, nn, prEntry.d_query)

        # Return the final result
        return nn.result_list()
    
    def range_search(self, query_obj, r=5):
        """
        Methode range_search:
            Algorithme de recherche K-plus proches voisins
        Args:
            query_obj: Q
            k: nombre de voisins

        Return the k objects the most similar to query_obj.
        Implementation of the k-Nearest Neighbor algorithm.
        Returns a list of the k closest elements to query_obj, ordered by
        distance to query_obj (from closest to furthest).
        If the tree has less objects than k, it will return all the
        elements of the tree.
        """
        if r == 0: return []
        pr = []
        # Initialiser pr avec le Noeud root et dmin et dquery a 0
        heappush(pr, PrEntry(self.root, 0, 0))
        # Un objet qui va contenir les k plus proches voisins
        # Un objet qui va contenir les k plus proches voisins
        nn = NN(100)
        while pr:
            # NextNode = ChooseNode(PR);
            prEntry = heappop(pr)
            if(prEntry.dmin > r):
                # best candidate is too far, we won't have better a answer
                # we can stop
                break
            # k_NN_NodeSearch(NextNode,Q,k);
            prEntry.tree.rangeSearch(query_obj, pr, nn, prEntry.d_query, r)
        # Return the final result
        return nn.result_list()






#_____________________________________ k NN array class __________________________________#

NNEntry = collections.namedtuple('NNEntry', 'obj dmax')

class NN(object):
    """
    a k-elements array, NN, which, at the end of execution,
    will contain the result.
    NN[i] = [oid(Oj),d(Oj, Q)], i = 1,...,k
    """
    def __init__(self, size):

        self.elems = [NNEntry(None, float("inf"))] * size
        #store dmax in NN as described by the paper
        #but it would be more logical to store it separately
        self.dmax = float("inf")

    def __len__(self):
        return len(self.elems)

    def search_radius(self):
        """The search radius of the knn search algorithm.
             aka dmax
             The search radius is dynamic.
        """
        return self.dmax

    def update(self, obj, dmax):
        """
        performs an ordered insertion in the NN array

        can change the search radius dmax: dk = NN_Update([_, dmax(T (Or))]);
        """
        # Si objet et null, update dmax
        if obj == None:
            # Internal node case
            self.dmax = min(self.dmax, dmax)
            return
        # Sinon, ajouter obj et ordonner
        # Leaf node case: We perform the insertion
        self.elems.append(NNEntry(obj, dmax)) # we have now k+1 elements
        # Sort
        for i in range(len(self)-1, 0, -1):
            if self.elems[i].dmax < self.elems[i-1].dmax:
                self.elems[i-1], self.elems[i] = self.elems[i], self.elems[i-1]
            else:
                break
        # we pop the last element so that we return to k
        # elements state
        self.elems.pop()

    # TODO: Update this to return oid alone!
    def result_list(self):
        """
        Retourne les objets de NN, les oid
        """
        result = list(map(lambda entry: entry.obj[0] if entry.obj else "None", self.elems))
        return result

#_____________________________________ PR class __________________________________
# PR is a queue of pointers to active sub-trees, i.e. sub-trees where
# qualifying objects can be found. With the pointer to (the root of) sub-tree T (Or),
# a lower bound, dmin(T(Or)), on the distance of any object in T(Or) from Q is also kept.

class PrEntry(object):
    def __init__(self, tree, dmin, d_query):
        """
        Constructor.
        arguments:
            tree: ptr(T(Or))
            dmin: dmin(T(Or))
            d_query: distance d to searched query object
        """
        self.tree = tree
        self.dmin = dmin
        self.d_query = d_query

    def __lt__(self, other):
        return self.dmin < other.dmin

    def __repr__(self):
        return "PrEntry(tree:%r, dmin:%r)" % (self.tree, self.dmin)




# <<<<<<<<<<<<<<<<< Done!
#_____________________________________ Classes Entry: Data (keys: Btree) ____________________________________#
class Entry(object): # Entry == Object
    """

    The leafs and internal nodes of the M-tree contain a list of instances of
    this class.

    The distance to the parent is None if the node in which this entry is
    stored has no parent (root).

    radius and subtree are None if the entry is contained in a leaf.
    Used in set and dict even tough eq and hash haven't been redefined

    Common attributs:
        obj: Oj or Or
        distance_to_parent: d(O,P(O))
    Leaf node attributs:
        not defined here : oid (Will be defined later for images)
    Internal node (non-leaf) attributs:
        radius: Covering radius r(Or).
        subtree: Pointer to covering tree T(Or).
    """
    def __init__(self, obj, distance_to_parent=None, radius=None, subtree=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent
        self.radius = radius
        self.subtree = subtree


#_____________________________________ Classes Nodes ___________________________________________#


class AbstractNode(object):
    """An abstract leaf of the M-tree.
    Concrete class are LeafNode and InternalNode

    We need to keep a reference to mtree so that we can know if a given node
    is root as well as update the root.

    We need to keep both the parent entry and the parent node (i.e. the node
    in which the parent entry is) for the split operation. During a split
    we may need to remove the parent entry from the node as well as adding
    a new entry to the node."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        #There will be an empty node (entries set is empty) when the tree
        #is empty and there only is an empty root.
        #May also be empty during construction (create empty node then add
        #the entries later).
        self.mtree = mtree
        self.parent_node = parent_node # Np
        self.parent_entry = parent_entry
        self.entries = set(entries) if entries else set() # NRO/NO

    def __repr__(self): # pragma: no cover
        #entries might be big. Only prints the first few elements
        entries_str = '%s' % list(islice(self.entries, 2))
        if len(self.entries) > 2:
            entries_str = entries_str[:-1] + ', ...]'

        return "%s(parent_node: %s, parent_entry: %s, entries:%s)" % (
            self.__class__.__name__,
            self.parent_node.repr_class() \
                if self.parent_node else self.parent_node,
            self.parent_entry,
            entries_str)

    def repr_class(self): # pragma: no cover
        return self.__class__.__name__ + "()"

    def __len__(self):
        return len(self.entries)

    @property
    def d(self):
        """
        Returns the distance function
        """
        return self.mtree.d

    def is_full(self):
        """
        Check if the node is full (with M entries)
        """
        return len(self) == self.mtree.max_nodes

    def is_empty(self):
        return len(self) == 0

    def is_root(self):
        return self is self.mtree.root

    def remove_entry(self, entry):
        """Removes the entry from this node
        Raise KeyError if the entry is not in this node
        """
        self.entries.remove(entry)

    def add_entry(self, entry):
        """Add an entry to this node.
        Raise ValueError if the node is full.
        """
        if self.is_full():
            raise ValueError("On peut pas ajoutee l'entree %  en un noeud plein " % str(entry))
        self.entries.add(entry)

    #TODO recomputes d(leaf, parent)!
    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        self.entries = new_entries
        self.parent_entry = new_parent_entry
        self.parent_entry.radius = self.covering_radius_for(self.parent_entry.obj)
        self._update_entries_distance_to_parent()


    #wastes d computations if parent hasn't changed.
    #How to avoid? -> check if the new routing_object is the same as the old
    # (compare id(obj) not obj directly to prevent == assumption about object?)
    def _update_entries_distance_to_parent(self):
        if self.parent_entry:
            for entry in self.entries:
                entry.distance_to_parent = self.d(entry.obj, self.parent_entry.obj)

    @abc.abstractmethod
    def add(self, obj):# pragma: no cover
        """Add obj into this subtree"""
        pass

    @abc.abstractmethod
    def covering_radius_for(self, obj):
        """Compute the radius needed for obj to cover the entries of this node.
        """
        pass

    @abc.abstractmethod
    def search(self, query_obj, pr, nn, d_parent_query):
        pass

    @abc.abstractmethod
    def rangeSearch(self, query_obj, pr, nn, d_parent_query, r):
        pass


#_____________________________________ Classe LeafNode __________________________________#
# Classe Represente un leaf noeud qui h'erite AbstractNode
class LeafNode(AbstractNode):
    """A leaf of the M-tree"""
    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        AbstractNode.__init__(self, mtree, parent_node, parent_entry, entries)

    def add(self, obj):
        distance_to_parent = self.d(obj, self.parent_entry.obj) if self.parent_entry else None
        new_entry = Entry(obj, distance_to_parent)
        # if N is not full just insert the new object
        if not self.is_full():
            # Store entry(On) in N
            self.entries.add(new_entry)
        else:
            # The node is at full capacity, then it is needed to do a new split in this level
            # Split(N,entry(On));
            split(self, new_entry, self.d)

    def covering_radius_for(self, obj):
        """
            Compute minimal radius for obj so that it covers all the objects
            of this node.
        """
        # Calcule le rayon couvrant
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj), self.entries))

    # k_NN_InternalNode_Search _____________________________________________________

    def could_contain_results(self, query_obj, search_radius, distance_to_parent, d_parent_query):
        if self.is_root():
            return True
        return abs(d_parent_query - distance_to_parent) <= search_radius

    def search(self, query_obj, pr, nn, d_parent_query):
        """
        Executes the K-NN query.

        Arguments:
            query_obj: Q
            pr: PR
            nn: NN
            d_parent_query:
        """
        # for each entry(Oj) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, nn.search_radius(),entry.distance_to_parent,d_parent_query):

                distance_entry_to_q = self.d(entry.obj, query_obj)

                if distance_entry_to_q <= nn.search_radius():
                    if(type(entry.obj) == int):
                        nn.update(entry.obj, distance_entry_to_q)
                    else:
                        print("[INFO] Voila les distances ", distance_entry_to_q)
                        nn.update(list(entry.obj), distance_entry_to_q)
    
    def rangeSearch(self, query_obj, pr, nn, d_parent_query, r):
        """
        Executes the K-NN query.

        Arguments:
            query_obj: Q
            pr: PR
            nn: NN
            d_parent_query:
        """
        # for each entry(Oj) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, r,entry.distance_to_parent,d_parent_query):

                distance_entry_to_q = self.d(entry.obj, query_obj)

                if distance_entry_to_q <= r:
                    if(type(entry.obj) == int):
                        nn.update(entry.obj, distance_entry_to_q)
                    else:
                        nn.update(list(entry.obj), distance_entry_to_q)



#_____________________________________ Classe InternalNode __________________________________#

# Classe represent un noeud interne qui h'erite AbstractNode
class InternalNode(AbstractNode):
    """An internal node of the M-tree"""
    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        AbstractNode.__init__(self, mtree, parent_node, parent_entry, entries)

    #TODO: apply optimization that uses the d of the parent to reduce the
    #number of d computation performed. cf M-Tree paper 3.3
    def add(self, obj):
        # put d(obj, e) in a dict to prevent recomputation
        # I guess memoization could be used to make code clearer but that is
        # too magic for me plus there is potentially a very large number of
        # calls to memoize
        dist_to_obj = {}
        for entry in self.entries:
            dist_to_obj[entry] = self.d(obj, entry.obj)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = [e for e in self.entries if dist_to_obj[e] <= e.radius]
            return min(valid_entries, key=dist_to_obj.get) if valid_entries else None

        def find_best_entry_minimizing_radius_increase():
            entry = min(self.entries, key=lambda e: dist_to_obj[e] - e.radius)
            entry.radius = dist_to_obj[entry]
            return entry

        entry = find_best_entry_requiring_no_covering_radius_increase() or find_best_entry_minimizing_radius_increase()
        entry.subtree.add(obj)

    def covering_radius_for(self, obj):
        """Compute minimal radius for obj so that it covers the radiuses
        of all the routing objects of this node
        """
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj) + e.radius,self.entries))

    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        AbstractNode.set_entries_and_parent_entry(self,new_entries,new_parent_entry)
        for entry in self.entries:
            entry.subtree.parent_node = self

    # k_NN_InternalNode_Search _____________________________________________________

    def could_contain_results(self, query_obj, search_radius, entry, d_parent_query):

        if self.is_root():
            return True

        return abs(d_parent_query - entry.distance_to_parent) <= search_radius + entry.radius

    def search(self, query_obj, pr, nn, d_parent_query):
        # for each entry(Or) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, nn.search_radius(), entry, d_parent_query):
                # Compute d(Or, Q);
                d_entry_query = self.d(entry.obj, query_obj)

                entry_dmin = max(d_entry_query - entry.radius, 0)

                if entry_dmin <= nn.search_radius():
                    heappush(pr, PrEntry(entry.subtree, entry_dmin, d_entry_query))
                    entry_dmax = d_entry_query + entry.radius
                    if entry_dmax < nn.search_radius():
                        nn.update(None, entry_dmax)
    

    def rangeSearch(self, query_obj, pr, nn, d_parent_query, r):
        # for each entry(Or) in N do:
        for entry in self.entries:
            if self.could_contain_results(query_obj, r, entry, d_parent_query):
                # Compute d(Or, Q);
                d_entry_query = self.d(entry.obj, query_obj)

                if d_entry_query <= r + entry.radius:
                    entry.subtree.rangeSearch(query_obj, pr, nn, d_parent_query, r)
   
    
    


#_____________________________________ SPLIT FUNCTION __________________________________#
# A lot of the code is duplicated to do the same operation on the existing_node
# and the new node :(. Could prevent that by creating a set of two elements and
# perform on the (two) elements of that set.
#TODO: Ugly, complex code. Move some code in Node/Entry?
def split(existing_node, entry, d):
    """
    Split existing_node N into two nodes N and N'.
    Adding entry to existing_node causes an overflow. Therefore we
    split existing_node into two nodes.

    Arguments:
        existing_node: N, full node to which entry should have been added
        entry: E, the added node. Caller must ensures that entry is initialized
               correctly as it would be if it were an effective entry of the node.
               This means that distance_to_parent must possess the appropriate
               value (the distance to existing_node.parent_entry).
        d: distance function.
    """
    # Get the reference to the M-Tree
    mtree = existing_node.mtree

    # The new routing objects are now all those in the node plus the new routing object
    all_entries = existing_node.entries | set((entry,)) # union


    # This part is already taken care of by the Entry class
    '''
        if N is not the root then
          {
             /*Get the parent node and the parent routing object*/
             let {\displaystyle O_{p}}O_{{p}} be the parent routing object of N
             let {\displaystyle N_{p}}N_{{p}} be the parent node of N
          }
    '''

    # This node will contain part of the objects of the node to be split */
    # Create a new node N'.
    # Type of the new node must be the same as existing_node
    # parent node, parent entry and entries are set later
    new_node = type(existing_node)(existing_node.mtree)

    #It is guaranteed that the current routing entry of the split node
    #(i.e. existing_node.parent_entry) is the one distance_to_parent
    #refers to in the entries (including the entry parameter).
    #Promote can therefore use distance_to_parent of the entries.

    # Promote two routing objects from the node to be split, to be new routing objects
    # Create new objects {\displaystyle O_{p1}}O_{{p1}} and {\displaystyle O_{p2}}O_{{p2}}.
    routing_object1, routing_object2 = mtree.promote(all_entries, existing_node.parent_entry, d)

    # Choose which objects from the node being split will act as new routing objects
    entries1, entries2 = mtree.partition(all_entries, routing_object1, routing_object2, d)

    message = "Error during split operation. All the entries have been assigned"
    message += "to one routing_objects and none to the other! Should never happen since at "
    message += "least the routing objects are assigned to their corresponding set of entries"
    assert entries1 and entries2, message


    # must save the old entry of the existing node because it will have
    # to be removed from the parent node later
    old_existing_node_parent_entry = existing_node.parent_entry

    # Setting a new parent entry for a node updates the distance_to_parent in
    # the entries of that node, hence requiring d calls.
    # promote/partition probably did similar d computations.
    # How to avoid recomputations between promote, partition and this?
    # share cache (a dict) passed between functions?
    # memoization? (with LRU!).
    #    id to order then put the two objs in a tuple (or rather when fetching
    #      try both way
    #    add a function to add value without computing them
    #      (to add distance_to_parent)

    #TODO: build_entry in the node method?
    # Store entries in each new routing object
    # Store in node N entries in N1 and in node N'entries in N2;
    existing_node_entry = Entry(routing_object1, None, None, existing_node)
    existing_node.set_entries_and_parent_entry(entries1, existing_node_entry)

    new_node_entry = Entry(routing_object2, None, None, new_node)
    new_node.set_entries_and_parent_entry(entries2, new_node_entry)


    if existing_node.is_root():
        # Create a new node and set it as new root and store the new routing objects Np
        new_root_node = InternalNode(existing_node.mtree)

        # Store Op1 and in Np
        existing_node.parent_node = new_root_node
        new_root_node.add_entry(existing_node_entry)
        # Store Op2 and in Np
        new_node.parent_node = new_root_node
        new_root_node.add_entry(new_node_entry)

        # Update the root
        mtree.root = new_root_node
    else:
        # Now use the parent routing object to store one of the new objects
        parent_node = existing_node.parent_node

        if not parent_node.is_root():
            # parent node has itself a parent, therefore the two entries we add
            # in the parent must have distance_to_parent set appropriately
            existing_node_entry.distance_to_parent = d(existing_node_entry.obj, parent_node.parent_entry.obj)
            new_node_entry.distance_to_parent = d(new_node_entry.obj, parent_node.parent_entry.obj)

        # Replace entry Op with entry Op1 in Np
        parent_node.remove_entry(old_existing_node_parent_entry)
        parent_node.add_entry(existing_node_entry)

        if parent_node.is_full():
            # If there is no free capacity then split the level up
            split(parent_node, new_node_entry, d)
        else:
            # The second routing object is stored in the parent
            # only if it has free capacity
            parent_node.add_entry(new_node_entry)
            new_node.parent_node = parent_node
