class IndexedMaxPQ:
    def __init__(self, N):
        self.N = N
        self.key = [None for _ in range(self.N)]
        self.pq = [None for _ in range(self.N + 1)]
        self.qp = [None for _ in range(self.N)]
        self.total = 0

    def insert(self, i, key):
        assert type(i) is int
        if i >= self.N:
            raise IndexError("Index is out of the range of IndexedMaxPQ.")
        if self.key[i] is not None:
            raise IndexError("Index is already in the IndexedMaxPQ.")
        self.total += 1
        self.key[i] = key
        self.pq[self.total] = i
        self.qp[i] = self.total
        self.__swim(self.total)

    def __swim(self, i):
        parent_i = i // 2

        while parent_i > 0:
            key = self.key[self.pq[i]]
            parent_key = self.key[self.pq[parent_i]]
            if parent_key > key:
                break
            self.pq[i], self.pq[parent_i] = self.pq[parent_i], self.pq[i]
            self.qp[self.pq[i]], self.qp[self.pq[parent_i]] = (
                self.qp[self.pq[parent_i]],
                self.qp[self.pq[i]],
            )
            i = parent_i
            parent_i = i // 2

    def deleteMax(self):
        if not self.isEmpty():
            out = self.key[self.pq[1]]
            self.key[self.pq[1]] = None
            self.qp[self.pq[1]] = None
            self.pq[1] = self.pq[self.total]
            self.qp[self.pq[1]] = 1
            self.pq[self.total] = None
            self.total -= 1
            self.__sink(1)
            return out, self.pq[1]
        raise IndexError("IndexedMaxPQ is empty")

    def __sink(self, i):
        child_i = i * 2
        if child_i <= self.total:
            key = self.key[self.pq[i]]
            child_key = self.key[self.pq[child_i]]
            other_child = child_i + 1
            if other_child <= self.total:
                other_child_key = self.key[self.pq[other_child]]
                if other_child_key > child_key:
                    child_i = other_child
                    child_key = other_child_key
            if child_key > key:
                self.pq[i], self.pq[child_i] = self.pq[child_i], self.pq[i]
                self.qp[self.pq[i]], self.qp[self.pq[child_i]] = (
                    self.qp[self.pq[child_i]],
                    self.qp[self.pq[i]],
                )
                self.__sink(child_i)

    def isEmpty(self):
        return self.total == 0

    def decreaseKey(self, i, key):
        if i < 0 or i >= self.N:
            raise IndexError("Index i is not in the range")
        if self.key[i] is None:
            # print("Index i is not in the IndexedMaxPQ")
            return
        assert type(i) is int
        assert key < self.key[i]
        self.key[i] = key
        self.__sink(self.qp[i])

    def increaseKey(self, i, key):
        if i < 0 or i >= self.N:
            raise IndexError("Index i is not in the range")
        if self.key[i] is None:
            raise IndexError("Index i is not in the IndexedMaxPQ")
        assert type(i) is int
        assert key > self.key[i]
        self.key[i] = key
        self.__swim(self.qp[i])


# Create a Maximum Indexed Priority Queue
max_pq = IndexedMaxPQ(10)

# Insert elements
max_pq.insert(1, 15)
max_pq.insert(2, 25)
max_pq.insert(3, 10)

# Get and remove the maximum element
print(max_pq.pq[1])
print(max_pq.deleteMax())  # Output: 25

# Increase key for an index
max_pq.increaseKey(3, 30)

# Peek the maximum element
print(max_pq.key[max_pq.pq[1]])  # Output: 30
print(max_pq.pq[1])
# Check if the queue is empty
print(max_pq.isEmpty())  # Output: False
