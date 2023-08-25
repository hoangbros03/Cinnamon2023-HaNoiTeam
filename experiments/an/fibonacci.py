import numpy as np
class Fibonacci():
    def __init__(self, first: int, second: int, num: int):
        self.first = first
        self.second = second
        self.num = num
    
    def forward(self):
        i = 2
        series = [self.first, self.second]
        prev = self.first
        curr = self.second
        while i < self.num:
            # curr = prev + curr
            # prev = curr - prev
            prev, curr = curr, prev + curr
            series.append(curr)
            i = i + 1
        return series
        
# s = []
i = 0
while i < 100:
    a = np.random.randint(1, 20)
    b = np.random.randint(a, 40)
    # s.append(Fibonacci(a, b, 20).forward())
    print(Fibonacci(a, b, 20).forward())
    i = i+1
# for i in s:
#     print(i)

# print(s)
        

