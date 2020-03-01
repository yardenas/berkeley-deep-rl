import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
        sess,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        return np.random.uniform(
            self.low,
            self.high,
            (num_sequences, horizon, self.ac_dim)
        )

    def get_action(self, obs):
        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)
        predicted_rewards_per_ens = []
        for model in self.dyn_models:
            if len(obs.shape)>1:
                observations = obs
            else:
                observations = obs[None]
            observations = np.tile(obs, (self.N, 1))
            cumulative_rewards = np.zeros(self.N)
            for t in range(self.horizon):
                rewards, dones = self.env.get_reward(observations, candidate_action_sequences[:, t, :])
                cumulative_rewards += rewards
                observations = model.get_prediction(
                    observations,
                    candidate_action_sequences[:, t, :],
                    self.data_statistics
                )
            predicted_rewards_per_ens.append(cumulative_rewards)
        predicted_rewards = np.mean(predicted_rewards_per_ens, axis=0)
        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards)
        best_action_sequence = candidate_action_sequences[best_index, ...]
        action_to_take = best_action_sequence[0]
        return action_to_take[None]
