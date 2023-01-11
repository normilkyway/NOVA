'''
Date: 1/2/2023
Program: NOVA Random Value Generator
    i. 3+ Data Modalities 
        a. Rotor Bearing Systems 
        b. TODO
        c. TODO
    ii. Simulate Data Realistically
        a. Normal Distribution of values with haphazard anomalies
        b. https://bitperfect.at/en/blog/simulation-von-sensordaten 
'''
import random
import matplotlib.pyplot as plt
import json

class Simulator:
    def __init__(self, seed, mean, std_dev, err_rate, err_len, smin, smax):
        self.mean = mean
        self.std_dev = std_dev
        self.step_sz = self.std_dev / 10
        self.val = self.mean - random.random()
        self.default_err_rate = err_rate
        self.default_err_len = err_len
        self.current_err_rate = err_rate
        self.current_err_len = err_len
        self.min = smin
        self.max = smax
        self.is_curr_err = False

        self.last_none_err_val = 0.0
        self.factors = [-1, 1]

        self.val_cnt = 0
        self.err_cnt = 0

        random.seed(seed)

    def calc_next_val(self):
        self.current_err_len -= 1
        if self.current_err_rate == self.current_err_len and self.current_err_rate < 0.01:
            self.current_err_rate = 0.01
        nextIsError = random.random() < self.current_err_rate
        if nextIsError:
           if not self.is_curr_err:
               self.last_none_err_val = self.val
           self.new_err_val()
        else:
           self.new_val()
           if self.is_curr_err:
               self.is_curr_err = False
               self.current_err_rate = self.default_err_rate
               self.current_err_len = self.default_err_len
        return self.val

    def new_val(self):
        self.val_cnt += 1
        delta_val = random.random() * self.step_sz
        factor = self.factors[0 if random.random() < 0.5 else 1]
        if self.is_curr_err:
            self.val = self.last_none_err_val
        self.val += delta_val * factor

    def new_err_val(self):
        self.err_cnt += 1
        if not self.is_curr_err:
            if self.val < self.mean:
                self.val = random.random() * (self.mean - 3 * self.std_dev - self.min) + self.min
            else:
                self.val = random.random() * (self.max - self.mean - 3 * self.std_dev) + self.mean + self.std_dev
        else:
            delta_val = random.random() * self.step_sz
            factor = self.factors[0 if random.random() < 0.5 else 1]
            self.val += delta_val * factor()

'''
data = []
simulation = Simulator(seed=random.randint(0, 1e99), mean=20, std_dev=5, err_rate=0.01, err_len=4.21, smin=0, smax=40) 
#Simulator(random.randint(0, 1e99), 20, 5, 0.01, 4.21, 0, 40)
for i in range(10_000):
    data.append(simulation.calc_next_val())

cnt = str(input('Enter test run #: '))
filename = 'data_' + cnt + '.txt'
with open(filename, 'w') as f:
    f.write(json.dumps(data))
    print(filename + 'successfully loaded...')

plt.plot(data)
plt.xlabel('epoch')
plt.ylabel('rotor speed')
plt.show()
print('Complete')
'''