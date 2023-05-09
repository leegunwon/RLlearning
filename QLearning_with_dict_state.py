import random
import numpy as np
import copy

class GridWorld():
    def __init__(self):
        self.x_history = []
        self.counter = 0

    def step(self, a):
        # 1번 액션: 왼쪽, 2번 액션: 오른쪽
        if a == 0:
            reward = -1
        elif a == 1:
            reward = +1

        self.x_history.append(a)
        self.counter += 1

        done = self.is_done()

        return self.x_history, reward, done

    def is_done(self):
        if self.counter == 6:
            return True
        else:
            return False

    def reset(self):
        self.x_history = []
        self.counter = 0
        return []


class QAgent():
    def __init__(self):
        self.key = [] 
        for i in range(2):
          self.key.append(tuple([i]))
          for j in range(2):
            self.key.append(tuple([i, j]))
            for k in range(2):
              self.key.append(tuple([i, j, k]))
              for l in range(2):
                self.key.append(tuple([i, j, k ,l]))
                for m in range(2):
                  self.key.append(tuple([i, j, k ,l, m]))
                  for n in range(2):
                    self.key.append(tuple([i, j, k ,l, m, n]))

        self.q_table = {k : [0, 0] for k in self.key} 
        self.q_table[()] = [0, 0]
        self.q_table[(0, 1, 0, 1, 0, 0)][0] = 1000
        self.eps = 0.9

    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다
        x = tuple(s)

        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 1)
        else:
            action_val = self.q_table[x]
            action = np.argmax(action_val)

        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x = tuple(s)
        x_prime = tuple(s_prime)


        # Q러닝 업데이트 식을 이용

        self.q_table[x][a] = self.q_table[x][a] + 0.1 * (r + np.amax(self.q_table[x_prime]) - self.q_table[x][a])


    def anneal_eps(self):
        self.eps -= 0.005  # Q러닝에선 epsilon 이 좀더 천천히 줄어 들도록 함.
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        key = list(self.q_table.keys())
        key.sort()
        for data in key:
          print(f"{data} : {self.q_table[data]}")


def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(10000):
        done = False
        s = env.reset()

        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s, a, r, s_prime))
            s = copy.deepcopy(s_prime)
        
        agent.anneal_eps()
    agent.show_table()


if __name__ == '__main__':
    main()
