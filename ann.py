from math import exp


class Node:
    def __init__(self, label, bias=None):
        self.label = label
        self.bias = bias
        self.error = None
        self.val = None
        self.edgeIn = []
        self.edgeOut = []


class Edge:
    def __init__(self, label, weight, node1, node2):
        self.label = label
        self.weight = weight
        self.node1 = node1
        self.node2 = node2


class ANN:
    def __init__(self, nodes, numInput, numOutput, l):
        self.nodes = nodes
        self.numInput = numInput
        self.numOutput = numOutput
        self.l = l
        self.edgesUpdated = []

    def train(self, inputs, outputs):
        self.forwardPropagate(inputs)
        self.backPropagate(outputs)
        self.printGraphInfo()
        self.saveGraphandReset()

    def forwardPropagate(self, inputs):
        # Set input values
        for i in range(self.numInput):
            self.nodes[i].val = inputs[i]

        # Set remaining node values
        for node in self.nodes[self.numInput:]:
            I = sum([e.weight * e.node1.val for e in node.edgeIn]) + node.bias
            node.val = 1 / (1 + exp(-I))

    def backPropagate(self, outputs):
        # Get output node errors
        numNodes = len(self.nodes)
        for i in range(numNodes - self.numOutput, numNodes):
            node = self.nodes[i]
            node.error = node.val * (1 - node.val) * (outputs[i - numNodes + self.numOutput] - node.val)
            # Update weights and prop
            self.prop(node)

    def prop(self, node):
        # Update bias
        if node.bias is not None:
            node.bias += self.l * node.error

        # Update weights
        for edge in node.edgeIn:
            edge.weight += self.l * node.error * edge.node1.val
            self.edgesUpdated.append(edge)
            nodesDone = True
            for e in edge.node1.edgeOut:
                if e not in self.edgesUpdated:
                    nodesDone = False
            if nodesDone:
                err = sum([e.weight * e.node2.error for e in edge.node1.edgeOut])
                edge.node1.error = edge.node1.val * (1 - edge.node1.val) * err
                self.prop(edge.node1)

    def saveGraphandReset(self):
        for node in self.nodes:
            node.error = None
            node.val = None
        self.edgesUpdated = []

    def printGraphInfo(self):
        print('\nBiases')
        print('--------------------------------------')
        for node in self.nodes:
            print('Node ' + node.label + ': ' + str(node.bias))

        print('\nEdge Weights')
        print('--------------------------------------')
        for edge in self.edgesUpdated:
            print('w' + edge.label + ': ' + str(edge.weight))


def createANN(l):
    # Input neuron
    O1 = Node('1')
    O2 = Node('2')

    # Hidden Neuron
    O3 = Node('3', bias=0.1)
    O4 = Node('4', bias=0.2)
    O5 = Node('5', bias=0.5)

    # Output Neurons
    O6 = Node('6', bias=-0.1)
    O7 = Node('7', bias=0.6)

    # Edges
    E13 = Edge('13', 0.1, O1, O3)
    E14 = Edge('14', 0, O1, O4)
    E15 = Edge('15', 0.3, O1, O5)

    E23 = Edge('23', -0.2, O2, O3)
    E24 = Edge('24', 0.2, O2, O4)
    E25 = Edge('25', -0.4, O2, O5)

    E36 = Edge('36', -0.4, O3, O6)
    E37 = Edge('37', 0.2, O3, O7)

    E46 = Edge('46', 0.1, O4, O6)
    E47 = Edge('47', -0.1, O4, O7)

    E56 = Edge('56', 0.6, O5, O6)
    E57 = Edge('57', -0.2, O5, O7)

    # Connect Graph
    O1.edgeOut = [E13, E14, E15]
    O2.edgeOut = [E23, E24, E25]

    O3.edgeIn = [E13, E23]
    O3.edgeOut = [E36, E37]

    O4.edgeIn = [E14, E24]
    O4.edgeOut = [E46, E47]

    O5.edgeIn = [E15, E25]
    O5.edgeOut = [E56, E57]

    O6.edgeIn = [E36, E46, E56]
    O7.edgeIn = [E37, E47, E57]

    nodes = [O1, O2, O3, O4, O5, O6, O7]
    return ANN(nodes, 2, 2, l)


if __name__ == '__main__':
    ann = createANN(0.1)
    ann.train([0.6, 0.1], [1, 0])
    ann.train([0.2, 0.3], [0, 1])
