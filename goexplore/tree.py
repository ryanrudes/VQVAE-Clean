class TreeNode:
    def __init__(self, root, code):
        self.root = root
        self.children = {}

    def add(self, action, code):
        node = TreeNode(self, code)
        self.children[action] = node
        return node

    def has(self, action):
        return action in self.children

    def get(self, action):
        return self.children[action]

class LinkedTree:
    def __init__(self, root_code):
        self.root = TreeNode(None, root_code)
        self.node = self.root

    def act(self, action, code):
        if self.node.has(action):
            self.node = self.node.get(action)
        else:
            self.node = self.node.add(action, code)

    def set(self, node):
        self.node = node
