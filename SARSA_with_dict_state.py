import random
import numpy as np
import copy
import itertools
import plotly.express as px

score = [[0, 0], [0, 0],[0, 0],[0, 0], [0, 0], [0, 0]]

class GridWorld():
    def __init__(self):
        self.x_history = []  # 에이전트의 행동을 기록하는 리스트
        self.counter = 0     # 에이전트가 몇 번 움직였는지 기록하는 변수

    def step(self, a):
        # 1번 액션: 왼쪽, 2번 액션: 오른쪽
        if a == 0:
            if self.x_history == [0, 1, 0, 1, 0]:
                reward = 1000
            else:
                reward = -1
                
        elif a == 1:
            reward = +1

        self.x_history.append(a)
        self.counter += 1


        done = self.is_done()

        return self.x_history, reward, done

    def is_done(self):
        if self.counter == 6:      # 6번 움직이면 끝
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
        for i in range(7):
            self.key += list(itertools.product([0, 1], repeat=i))    # 딕셔너리의 키에 해당하는 리스트

        self.q_table = {k: [0, 0] for k in self.key} # 딕셔너리의 초기값을 0으로 설정
        self.eps = 0.9

    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다
        x = tuple(s)   # 리스트 s를 튜플로 변환 (딕셔너리의 경우 immutable한 자료형만 키로 사용 가능)

        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 1)
        else:
            action_val = self.q_table[x]
            action = np.argmax(action_val)

        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x = tuple(s)            # 리스트 s를 튜플로 변환 (딕셔너리의 경우 immutable한 자료형만 키로 사용 가능)
        x_prime = tuple(s_prime)
        a_prime = self.select_action(s_prime)
        
        # Q러닝 업데이트 식을 이용
        self.q_table[x][a] = self.q_table[x][a] + 0.1 * (r + self.q_table[x_prime][a_prime] - self.q_table[x][a])

    def anneal_eps(self):
        self.eps -= 0.005
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        key = list(self.q_table.keys())    
        key.sort()
        for data in key:
            print(f"{data} : 액션 {np.argmax(self.q_table[data])}")


def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(3000):
        done = False
        s = env.reset()

        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s, a, r, s_prime))
            s = copy.deepcopy(s_prime)

        agent.anneal_eps()
       #  print(s)
    # agent.show_table()

    score[0][0] += agent.q_table[()][0]
    score[1][1] += agent.q_table[(0,)][0]
    score[2][0] += agent.q_table[(0, 1)][0]
    score[3][0] += agent.q_table[(0, 1, 0)][0]
    score[4][0] += agent.q_table[(0, 1, 0, 1)][0]
    score[5][0] += agent.q_table[(0, 1, 0, 1, 0)][0]
    score[0][1] += agent.q_table[()][1]
    score[1][1] += agent.q_table[(0,)][1]
    score[2][1] += agent.q_table[(0, 1)][1]
    score[3][1] += agent.q_table[(0, 1, 0)][1]
    score[4][1] += agent.q_table[(0, 1, 0, 1)][1]
    score[5][1] += agent.q_table[(0, 1, 0, 1, 0)][1]




if __name__ == '__main__':
    for i in range(10):
        main()
    print(score)

    fig = px.bar(x=['', '0', '01', '010', '0101', '01010'], y=[score[0][0]/10 - score[0][1]/10, score[1][0]/10- score[1][1]/10, score[2][0]/10- score[2][1]/10, score[3][0]/10 -score[3][1]/10, score[4][0]/10 - score[4][1]/10, score[5][0]/10- score[5][1]/10], color=['', '0', '01', '010', '0101', '01010'])
    fig.show()
