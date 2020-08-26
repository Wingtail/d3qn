import numpy as np
from queue import Queue

class SegmentTree_Node():
    def __init__(self, value, priority, parent):
        self.value = value
        self.priority = priority

        self.child_nodes = []
        self.parent = parent

    def add_node(self, value, priority):
        if self.value is None:
            if self.parent is None:
                self.value = value
                self.priority = priority
                return [self]
            else:
                print("Value is None, wtf")
                assert self.value is not None
        else:
            children = self.create_children(value, priority)
            self.update_ancestry_branch()
            return children

    def self_destruct(self):
        child_1, child_2 = self.parent.child_nodes
        if child_1.value != self.value:
            self.parent.value = child_1.value
            self.parent.priority = child_1.priority
        else:
            self.parent.value = child_2.value
            self.parent.priority = child_2.priority

        self.parent.child_nodes = []

    def update_ancestry_branch(self):
        queue = Queue()
        queue.put(self)
        while(not queue.empty()):
            node = queue.get()
            self.update_parent_priority(node)
            if node.parent is not None:
                queue.put(node.parent)

    def create_children(self, value, priority):
        print("You didn't implement create_children!!")
        assert False
        pass

    def update_parent_priority(self):
        print("You didn't implement update_parent_priority!!")
        assert False
        pass

class SumTree_Node(SegmentTree_Node):
    def __init__(self, value, priority, parent):
        super(SumTree_Node, self).__init__(value, priority, parent)

    def add_node(self, value, priority):
        new_child_1 = None
        new_child_2 = None
        if self.value is None:
            if self.parent is None:
                self.value = value
                self.priority = priority
            else:
                print("Value is None, wtf")
                assert self.value is not None
        else:
            new_child_1 = SumTree_Node(self.value, self.priority, self)
            new_child_2 = SumTree_Node(value, priority, self)
        super().add_node(new_child_1, new_child_2)

    def update_ancestry_branch(self):
        queue = Queue()
        queue.put(self)
        while(not queue.empty()):
            node = queue.get()
            node.priority = sum([child.priority for child in node.child_nodes])
            if node.parent is not None:
                queue.put(node.parent)

class DynamicSegmentTree_Node(SegmentTree_Node):
    def __init__(self, value, priority, parent):
        super(DynamicSegmentTree_Node, self).__init__(value, priority, parent)

    def create_children(self, value, priority):
        new_child_1 = DynamicSegmentTree_Node(self.value, self.priority, self)
        new_child_2 = DynamicSegmentTree_Node(value, priority, self)
        self.child_nodes = [new_child_1, new_child_2]
        self.value = None
        self.priority = None
        return self.child_nodes

    def update_parent_priority(self, node):
        if len(node.child_nodes)>0:
            sum_priority = sum([child.priority[0] for child in node.child_nodes])
            min_priority = min([child.priority[1] for child in node.child_nodes])
            max_priority = max([child.priority[2] for child in node.child_nodes])
            node.priority = [sum_priority, min_priority, max_priority]


class DynamicSegmentTree():
    def __init__(self, eps=1e-9, alpha=0.4, beta=0.4):
        '''
        Binary Tree
        '''
        self.root_node = DynamicSegmentTree_Node(None, None, None)
        self.add_node_queue = Queue()
        self.add_node_queue.put(self.root_node)

        self.max_weight = None
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

        self.leaf_nodes = {}

    def get_sum_priority(self):
        if self.root_node.priority is None:
            return None
        else:
            return self.root_node.priority[0]

    def get_min_priority(self):
        if self.root_node.priority is None:
            return None
        else:
            return self.root_node.priority[1]

    def get_max_priority(self):
        if self.root_node.priority is None:
            return None
        else:
            return self.root_node.priority[2]

    def get_importance_weight(self, priority):
        prob = priority / self.get_sum_priority()
        return (len(self.leaf_nodes) * prob)**(-self.beta)

    def get_max_importance_weight(self):
        min_priority = self.get_min_priority()
        return self.get_importance_weight(min_priority)

    def select_node(self, value):
        current_node = self.root_node

        while len(current_node.child_nodes) > 0:
            if value < current_node.child_nodes[0].priority[0]:
                current_node = current_node.child_nodes[0]
            else:
                value -= current_node.child_nodes[0].priority[0]
                current_node = current_node.child_nodes[1]

        norm_weight = self.get_importance_weight(current_node.priority[0]) / self.max_weight

        return current_node.value, norm_weight

    def sample(self, sample_size):
        values = np.random.uniform(low=0, high=self.root_node.priority[0], size=sample_size)
        node_values = list(map(self.select_node, values))

        index = np.zeros(len(node_values), dtype=np.int32)
        weights = np.zeros(len(node_values), dtype=np.float32)

        for i in range(len(node_values)):
            index[i] = node_values[i][0]
            weights[i] = node_values[i][1]

        return index, weights

    def update_node(self, value, priority=None):
        if priority is None:
            if self.get_max_priority() is None:
                priority = self.eps**self.alpha
            else:
                priority = self.get_max_priority()
        else:
            priority = (priority + self.eps)**self.alpha

        if value in self.leaf_nodes:
            leaf_node = self.leaf_nodes[value]
            leaf_node.priority = [priority, priority, priority]
            leaf_node.update_ancestry_branch()
        else:
            node = self.add_node_queue.get()
            children = node.add_node(value, [priority, priority, priority])

            for child in children:
                self.leaf_nodes[child.value] = child
                self.add_node_queue.put(child)

        self.max_weight = self.get_max_importance_weight()

    def remove_node(self, value):
        self.leaf_nodes[value].self_destruct()
        del self.leaf_nodes[value]
