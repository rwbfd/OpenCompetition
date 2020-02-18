# coding = 'utf-8'

class GradientUpdate:
    def __init__(self):
        self.s = list()
        self.gamma_a = None

    def re_init(self):
        self.s = list()
        self.gamma_a = None

    def get_gradient_for_single_observation(self):
        self.re_init()
        trajectories = self.get_trajectory()  # TODO: This needs to be implemented
        
    def get_trajectory(self):  # TODO: This needs to be implemented.
        pass

    def determine_budget(self):  # TODO: this needs to be implemented
        pass

    def get_a_and_d(self):  # TODO: This needs to be implemented
        pass
