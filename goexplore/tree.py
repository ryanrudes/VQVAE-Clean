from sys import getsizeof as size

class TreeNode:
    def __init__(self, root, code, action):
        self.root = root
        self.action = action
        self.children = {}

    def __sizeof__(self):
        total = 0
        stack = [self.root, self.action, self.children]
        while stack:
            object = stack.pop()
            if isinstance(object, TreeNode):
                stack.extend([id(object.root), object.action, object.children])
            elif isinstance(object, dict):
                for key, value in object.items():
                    stack.extend([key, value])
            else:
                total += size(object)
        return total

    def add(self, action, code):
        node = TreeNode(self, code, action)
        self.children[action] = node
        return node

    def has(self, action):
        return action in self.children

    def get(self, action):
        return self.children[action]

class LinkedTree:
    def __init__(self, root_code):
        self.root = TreeNode(None, root_code, None)
        self.node = self.root

    def __sizeof__(self):
        return size(self.root)

    def act(self, action, code):
        if self.node.has(action):
            self.node = self.node.get(action)
        else:
            self.node = self.node.add(action, code)

    def set(self, node):
        self.node = node

    def get_trajectory(self, temp=None):
        if temp is None:
            temp = self.node
        trajectory = []
        while not temp.action is None:
            trajectory.append(temp.action)
            temp = temp.root
        return trajectory

    def size(self):
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += len(node.children)
            stack.extend(list(node.children.values()))
        return count
