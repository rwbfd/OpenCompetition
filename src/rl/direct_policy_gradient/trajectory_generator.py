# coding = 'utf-8'
from scipy.stats import gompertz
from scipy.stats import gumbel_r
import asyncio
import math

DELTA = 0.00001  # A small number to avoid log(0


class TrajectoryGenerator:
    """
    This class performs the trajectory generation.
    """

    def __init__(self, config, env, action_space):
        self.config = config
        self.env = env
        self.action_space = action_space
        self.queue = asyncio.Queue()

    def reset_queue(self):
        self.queue = asyncio.Queue()

    def _generate_single_trajectory(self, gamma, epsilon, state_reward_tree, policy_net, pi_theta, r,
                                    is_terminal):
        """

        :param gamma: the gamma as eq 4.
        :param epsilon
        :param state_reward_tree: The state reward tree
        :return:
        """
        self.reset_queue()
        q = self.reset_queue()
        s = state_reward_tree.get_s()
        r = list()
        q.push(0, self.action_space, float(TrajectoryGenerator._get_gumbel(1)))  # TODO: This needs to be checked
        while not q.empty():  # TODO: This needs to be checked.
            a_tilde, b, g = q.pop()
            a = policy_net.get_action(s)  # TOOD: need to implement the policy net interface
            new_state, new_reward = self.env.step(a, s)
            s = s.append(new_state)
            r = r.append(new_reward)
            if not b.difference({a}).empty():
                mu = math.log(pi_theta(r(a_tilde, b.difference({a}), s)) + DELTA)  # TODO: This needs to be checked
                g_prime = TrajectoryGenerator._get_truncate_gumbel(mu)
                q.push(a_tilde, b.dffferece({a}), g_prime)
                a_tilde = a_tilde.append(a)
            if is_terminal(s):  # TODO: Might need to add a budget
                a_tilde = a_tilde.append(a)
                return a_tilde, epsilon * r(a_tilde, s)  # TODO: Needs to check whether this s is correct
            else:
                q.append(a_tilde, self.action_space, g)

    @staticmethod
    def _get_gumbel(size=0):
        if size == 0:
            return gumbel_r.rvs()
        else:
            return gumbel_r.rvs(size)

    @staticmethod
    def _get_truncate_gumbel(mu, size=0):
        if size == 0:
            return gompertz.rvs(mu)
        else:
            return gompertz.rvs(mu, size)
