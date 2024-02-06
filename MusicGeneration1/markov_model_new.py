from random import choice

class Node:
    def __init__(self, state=None, next_node=None):
        self.state = state
        self.next = next_node

class Memory:
    """ Memory circular buffer implemented as a linked list based queue for MarkovModel's higher order functionality """

    def __init__(self, order=1, orderadditive=0, starterlength=400, lengthadd=1, lengthnegate=1):
        self.head = None
        self.tail = None
        self.order = order
        self.orderadditive = orderadditive
        self.lengthadd = lengthadd
        self.lengthnegate = lengthnegate
        #self.length = 0
        self.trueorder = order + orderadditive
        self.length = starterlength

    def enqueue(self, state=None):
        """ Add a node to the start of the queue with state. Length will never exceed order """
        print('enqueueing')
        self.length += self.lengthadd
        if self.length > self.trueorder:
            self.dequeue()
        if self.length == self.trueorder:
            print('length = trueorder')
        if self.head is None:
            self.head = Node(state=state)
            self.tail = self.head
        else:
            new_node = Node(state=state)
            self.tail.next = new_node
            self.tail = new_node

    def dequeue(self):
        """ Remove a node from the end of the queue. Returning is not necessary due to implementation """
        print('dequeueing')
        if self.head is not None:
            if self.length < self.trueorder:
                self.enqueue()
            self.length -= self.lengthnegate
            self.head = self.head.next
            if self.head is None:
                self.tail = None

    def clear(self):
        """ Clear memory """
        self.head = None
        self.tail = None
        self.length = 0

    def serialize(self):
        """ Serializes memory queue in the form of a tuple of strings, making it hashable """
        return tuple(str(node.state) for node in self)

    def __len__(self):
        return self.length

    def __iter__(self):
        return MemoryIterator(self.head)

class MemoryIterator:
    """ Make memory queue iterable """

    def __init__(self, head):
        self.current_node = head

    def __next__(self):
        if self.current_node is not None:
            prev_node = self.current_node
            self.current_node = prev_node.next
            return prev_node

        raise StopIteration

class MarkovModel(dict):
    """ Dictionary based nth order markov model """

    def __init__(self, midi_data=(), order=1, orderadditive=0, lengthadditive=1, lengthnegate=1, starterlength=400):
        self.memory = Memory(order=order, orderadditive=orderadditive, lengthadd=lengthadditive,
                             lengthnegate=lengthnegate, starterlength=starterlength)

        for message in midi_data:
            if isinstance(message, str) and message == 'START':
                self.memory.clear()
                self.memory.enqueue('START')
            else:
                self.add_state(message)

        self.memory.clear()
        self.memory.enqueue('START')

    def add_state(self, new_state):
        """ Add a state to MarkovModel and add new state to memory """
        current_state = self.memory.serialize()

        if current_state in self:
            self[current_state].append(new_state)
        else:
            self[current_state] = [new_state]

        self.memory.enqueue(new_state)

    def sample(self):
        """ Return generator that samples from MarkovModel until an end state is reached """
        starting_state = ('START',)
        while True:
            next_state = choice(self[starting_state])
            self.memory.enqueue(next_state)
            # Continue yielding until an end state is reached
            if not isinstance(next_state, str):
                yield next_state
            else:
                break
            starting_state = self.memory.serialize()
        self.memory.clear()
        self.memory.enqueue('START')