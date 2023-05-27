import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time

Ept_A_rent = 3
Ept_A_return = 3
Ept_B_rent = 4
Ept_B_return = 2
depot_max_car = 20
benefit_rent = 10
cost_move = 2
poisson_ub = 10
num_workers = 12

set_S = [(i,j) for i in range(21) for j in range(21)]
set_A = [(i,-i) for i in range(-5,6)]
pi_a_s = dict.fromkeys(set_S, (0,0))
vpi_s = dict.fromkeys(set_S, 0)
vpi_delta = 1e-3
gamma = 0.9


def poisson_poss(ept):
    return [(n, (ept ** n) / np.math.factorial(n) * (np.exp(-ept))) for n in range(0, poisson_ub+1)]

def calculate_s_r(s_a):
    s, a = s_a[0], s_a[1]
    # print(f'Generating state {s} action {a}')
    p_s_a = []

    for req_A_rent, prob_A_rent in poisson_poss(Ept_A_rent):
        for req_B_rent, prob_B_rent in poisson_poss(Ept_B_rent):
            for req_A_return, prob_A_return in poisson_poss(Ept_A_return):
                for req_B_return, prob_B_return in poisson_poss(Ept_B_return):
                    ### assume that num. returned cars is constant
                    # req_A_return, req_B_return = Ept_A_return, Ept_B_return

                    s_, r = [s[0],s[1]], 0

                    ### Move the cars during the night
                    s_[0] = min(s_[0] + a[0], depot_max_car)
                    s_[1] = min(s_[1] + a[1], depot_max_car)
                    r -= abs(a[0]) * cost_move

                    ### Rent cars
                    rent_suc_A = min(s_[0], req_A_rent)
                    rent_suc_B = min(s_[1], req_B_rent)
                    s_[0] -= rent_suc_A
                    s_[1] -= rent_suc_B
                    r += (rent_suc_A + rent_suc_B) * benefit_rent

                    ### Return cars
                    s_[0] = min(s_[0] + req_A_return, depot_max_car)
                    s_[1] = min(s_[1] + req_B_return, depot_max_car)

                    p_s_a.append((tuple(s_),r,prob_A_rent*prob_B_rent*prob_A_return*prob_B_return))
    
    return p_s_a

def generate_p():
    p_keys = [(s,a) for s in set_S for a in set_A if (0 <= a[0]+s[0] <= depot_max_car) and (0 <= a[1]+s[1] <= depot_max_car)]
    with Pool(num_workers) as p:
        p_values = p.map(calculate_s_r, p_keys)

    state_trans_p = dict(zip(p_keys,p_values))     
    return state_trans_p

def policy_evaluation():
    global vpi_s
    while(True):
        delta = 0
        for s in set_S:
            v = vpi_s[s]
            vpi_s[s] = sum([prob * (r + gamma * vpi_s[s_]) for s_, r, prob in state_trans_p[(s, pi_a_s[s])]])
            delta = max(delta, abs(vpi_s[s] - v))
        # print(f'Iter{iter_step} Policy Evaluation: Delta = {delta}')
        if delta < vpi_delta:
            return

def policy_improvement():
    global pi_a_s
    policy_stable = True
    for s in set_S:
        _a = pi_a_s[s]
        qpi_sa_dict = {}
        for a in set_A:
            if (0 <= a[0]+s[0] <= depot_max_car) and (0 <= a[1]+s[1] <= depot_max_car):
                qpi_sa = sum([prob * (r + gamma * vpi_s[s_]) for s_, r, prob in state_trans_p[s,a]])
                qpi_sa_dict[a] = qpi_sa
        pi_a_s[s] = max(qpi_sa_dict, key=qpi_sa_dict.get)

        if _a != pi_a_s[s]:
            policy_stable = False
        # print(f'Iter{iter_step} Policy Improvement: Policy_stable = {policy_stable}')

    return policy_stable

def plot_pi_a_s(iter_step):
    data_pi_a_s = np.zeros((21,21))
    for i in range(21):
        for j in range(21):
            data_pi_a_s[i,j] = pi_a_s[(i,j)][0]
    sns.heatmap(data_pi_a_s, cmap="YlGnBu")
    plt.title(f'Policy at Iteration {iter_step}')
    plt.savefig(f"./poisson_4.3_jack_rental_case_pi_a_s_iter{iter_step}.png",dpi=300,bbox_inches='tight',pad_inches=0)
    plt.close()

def plot_vpi_s():
    data_vpi_s = np.zeros((21,21))
    for i in range(21):
        for j in range(21):
            data_vpi_s[i,j] = vpi_s[(i,j)]
    sns.heatmap(data_vpi_s, cmap="RdYlBu")
    plt.title(f'Value Function (Converged)')
    plt.savefig(f"./poisson_4.3_jack_rental_case_vpi_s.png",dpi=300,bbox_inches='tight',pad_inches=0)
    plt.close()


if __name__ == '__main__':
    t0 = time.time()
    state_trans_p = generate_p()
    t1 = time.time()
    print(f'Data Initialized: Taking {t1-t0}s')

    iter_step = 1
    ### Policy Iteration
    while(True):
        plot_pi_a_s(iter_step)

        t_2_0 = time.time()
        policy_evaluation()
        t_2_1 = time.time()
        policy_stable = policy_improvement()
        t_2_2 = time.time()

        print(f'Iter {iter_step}: policy_evaluation {t_2_1-t_2_0}s && policy_stable {t_2_2-t_2_1}s')
        if policy_stable:
            plot_vpi_s()
            break

        iter_step += 1