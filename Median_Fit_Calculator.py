"""
Date: 1/2/2023
Program: NOVA Statistical Analysis ==> MedianHeap
    1. Median Line Analysis
        i. TC - O(NlogN), much faster than brute force O(N^2)

Helpful: 
StackOverFlow: https://stackoverflow.com/questions/68949041/running-median-in-olog-n-time-in-python
Wiki: https://en.wikipedia.org/wiki/Counting_sort (Counting Sort Methodology)
"""
import collections
import heapq
import math
from typing import Union, List, Counter
"""import json"""

class MedianHeap:
    def __init__(self):
        """ Data structure for holding numeric types, with O(log n) amortized insertion and deletion and
        constant worst case median finding.
        Maintains two heaps:
        a max heap with values smaller than (or eq. to) the median, and a
        min heap with values larger than (or eq. to) the median).
        We use lazy deletion from heaps (rebuilding them if more than ~50% of the heaps are deleted elements)
         to guarantee amortized insertion and deletion.

        The following class invariants are maintained before an external function call starts
        and after that function call ends:
            1. The front of each heap is either infinity (the empty heap sentinel), or a valid element
            2. All elements in self.max_heap have a value less than or equal to all elements in self.min_heap
            3. Size (i.e. undeleted elements) of self.min_heap is 0 or 1 plus the size of self.max_heap"""

        self.min_heap: List[Union[int, float]] = [math.inf]  # Add a sentinel value, to avoid repeated empty checks.
        self.max_heap: List[Union[int, float]] = [math.inf]  # on the left: all elements <= effective median

        self.total_real_elems: int = 0
        self.min_real_elems: int = 0
        self.max_real_elems: int = 0

        self.deleted: Counter[int, int] = collections.Counter()

        # If lazy deletion has caused our data structures to fill too much with deleted elements, trigger a rebuild
        # We rebuild if: (lazy_deletion_multiplier * #deleted) > total_size + lazy_deletion_constant.

        # By default, the multiplier is set at 50%, as is common for open-addressing hash tables which also use
        # lazy deletion. The constant can be increased based on performance needs.
        self.lazy_deletion_multiplier: int = 2
        self.lazy_deletion_constant: int = 500

    def insert(self, num: int) -> None:
        """Insert num into our MedianHeap. May not trigger a full rebuild. O(lg n) worst case time."""
        if not (-math.inf < num < math.inf):
            raise ValueError
        if self.total_real_elems == 0:
            heapq.heappush(self.min_heap, num)
            self.total_real_elems += 1
            self.min_real_elems += 1
            return None

        if num >= self.min_heap[0]:
            heapq.heappush(self.min_heap, num)
            self.min_real_elems += 1
        else:
            heapq.heappush(self.max_heap, -num)
            self.max_real_elems += 1

        self.total_real_elems += 1
        self._rebalance()
        return None

    def remove(self, num: int) -> None:
        """Change the status of one instance of 'num' from active to deleted. O(lg n) amortized, O(n) worst case time.
        num must be an active element in our data structure. May trigger a rebuild."""
        if num >= self.min_heap[0]:
            if num == self.min_heap[0]:
                heapq.heappop(self.min_heap)
            else:
                self.deleted[num] += 1
            self.min_real_elems -= 1
        else:
            if num == -self.max_heap[0]:
                heapq.heappop(self.max_heap)
            else:
                self.deleted[num] += 1
            self.max_real_elems -= 1

        self.total_real_elems -= 1
        self._rebalance()

    def _clean_min_heap_front(self) -> None:
        """While the front of the min_heap was already deleted, remove it from the min_heap"""
        while self.deleted[self.min_heap[0]] > 0:
            self.deleted[heapq.heappop(self.min_heap)] -= 1

    def _clean_max_heap_front(self) -> None:
        """While the front of the max_heap was already deleted, remove it from the max_heap"""
        while self.deleted[-self.max_heap[0]] > 0:
            self.deleted[-heapq.heappop(self.max_heap)] -= 1

    def _rebuild_fully(self) -> None:
        """To guarantee O(log n) amortized insertions and deletions with lazy deletions, we must detect when the number
        of removed elements still in our heap has grown too large: If so, perform an O(n) full rebuild of both heaps
        from scratch, clearing all previously removed elements."""

        # Rebuild heaps, trying to maintain size approximately based on the median
        approx_median: int = self.min_heap[0]

        new_min_heap: List[Union[int, float]] = [math.inf]
        new_max_heap: List[Union[int, float]] = [math.inf]

        for elem in self.max_heap:
            if self.deleted[-elem] > 0:
                self.deleted[-elem] -= 1
                continue
            elif math.isinf(elem):
                continue
            if -elem < approx_median:
                new_max_heap.append(elem)
            elif -elem > approx_median:
                new_min_heap.append(-elem)
            else:
                if len(new_min_heap) - len(new_max_heap) > 1:
                    new_max_heap.append(elem)
                else:
                    new_min_heap.append(-elem)

        for elem in self.min_heap:
            if self.deleted[elem] > 0:
                self.deleted[elem] -= 1
                continue
            elif math.isinf(elem):
                continue

            if elem < approx_median:
                new_max_heap.append(-elem)
            elif elem > approx_median:
                new_min_heap.append(elem)
            else:
                if len(new_min_heap) - len(new_max_heap) > 1:
                    new_max_heap.append(-elem)
                else:
                    new_min_heap.append(elem)

        self.min_heap = new_min_heap
        self.max_heap = new_max_heap

        heapq.heapify(self.min_heap)
        heapq.heapify(self.max_heap)

        self.deleted.clear()
        self.min_real_elems = len(self.min_heap) - 1
        self.max_real_elems = len(self.max_heap) - 1
        self.total_real_elems = self.min_real_elems + self.max_real_elems

        if not (0 <= (self.min_real_elems - self.max_real_elems) <= 1):
            self._rebalance()

    def _need_full_rebuild_check(self) -> bool:
        """Test whether our heaps have a larger fraction of removed elements than allowed"""
        total_size: int = len(self.min_heap) + len(self.max_heap)
        return (self.lazy_deletion_multiplier * (total_size - self.total_real_elems)
               > total_size + self.lazy_deletion_constant)

    def _rebalance(self):
        """ Restore the class invariants:
        1. Front of each heap is infinity (empty heap or sentinel), or a valid element
        2. All elements in self.max_heap have a value <= all elements in self.min_heap
        3. Size (i.e. undeleted elements) of self.min_heap - size of self.max_heap is 0 or 1"""

        if self._need_full_rebuild_check():
            self._rebuild_fully()
            return None

        self._clean_min_heap_front()
        self._clean_max_heap_front()

        while -self.max_heap[0] > self.min_heap[0]:
            if self.min_real_elems - self.max_real_elems <= -1:  # Prefer deleting from max_heap
                self.max_real_elems -= 1
                self.min_real_elems += 1
                heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
                self._clean_max_heap_front()

            else:  # Prefer deleting from min_heap
                self.max_real_elems += 1
                self.min_real_elems -= 1
                heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
                self._clean_min_heap_front()

        while self.min_real_elems - self.max_real_elems <= -1:  # Need to reduce size of max_heap
            self.max_real_elems -= 1
            self.min_real_elems += 1
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
            self._clean_max_heap_front()  # Removing front of a heap may place a deleted element in front

        while self.min_real_elems - self.max_real_elems > 1:  # Need to reduce size of min_heap
            self.max_real_elems += 1
            self.min_real_elems -= 1
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
            self._clean_min_heap_front()  # Removing front of min_heap may place a deleted element in front

        return None

    def calculate_median(self) -> float:
        """Calculate the median in constant time: the median element(s) are always in a heap's front."""
        if self.total_real_elems == 0:
            raise IndexError

        if self.total_real_elems % 2 == 0:
            return (self.min_heap[0] - self.max_heap[0]) / 2.0
        else:
            return self.min_heap[0]

"""
'''Testing MedianHeap dataset'''
data = []
cnt = "1" #str(input('Access id#: '))
filename = 'data_' + cnt + '.txt'
with open(filename, 'r') as f:
    data = json.loads(f.read())
    # print(data)
    print('Successfully loaded ' + filename + '... ')
median_list = []
md_alg = MedianHeap()
for i in range(len(data)):
    md_alg.insert(data[i])    
    median_list.append(md_alg.calculate_median())
    
print(len(median_list))
print(len(data))
print(median_list)
"""