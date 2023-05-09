import random
import numpy as np


class GridWorld():
    def __init__(self):
        self.result = []
        self.x_history = [i for i in range(6)]
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
        if len(self.x_history) == 6:
            return True
        else:
            return False

    def reset(self):
        self.x_history = [i for i in range(6)]
        self.counter = 0
        return self.x_history


class QAgent():
    def __init__(self):
        self.q_table = {}  # 마찬가지로 Q 테이블을 0으로 초기화
        self.eps = 0.9

    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다
        num = 0
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
        
        if x not in  self.q_table.keys():
          self.q_table[x] = [0, 0]
        
        if x_prime not in self.q_table.keys():
          if len(x_prime) == 6:
            self.q_table[x_prime] = 0

        # Q러닝 업데이트 식을 이용
        self.q_table[x][a] = self.q_table[x][a] + 0.1 * (
                    r + np.amax(self.q_table[x_prime]) - self.q_table[x][a])

    def anneal_eps(self):
        self.eps -= 0.01  # Q러닝에선 epsilon 이 좀더 천천히 줄어 들도록 함.
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        q_lst = self.q_table.tolist()
        data = [0,0,0,0,0,0]
        for idx in range(len(q_lst)):
            row = q_lst[idx]
            action = np.argmax(row)
            data[idx] = action
        print(data)


def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(100):
        done = False
        s = env.reset()

        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s, a, r, s_prime))
            s = s_prime
        agent.anneal_eps()
        if env.x_history == [0,1,0,1,0,0]:
          agent.q_table[(0,1,0,1,0,0)] += 1000


if __name__ == '__main__':
    main()
