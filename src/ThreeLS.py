
# Optimal Control and Reinforcement Learning tutorial for
# three-level population transfer
# Copyright (C) 2022 Luigi Giannelli

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pickle

import matplotlib.pyplot as plt
import numpy as np
from qutip import (Options, basis, expect, ket2dm, liouvillian, mesolve,
                   operator_to_vector, vector_to_operator)
from scipy.optimize import minimize
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tqdm.auto import trange

opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
# opts = Options(atol=1e-13, rtol=1e-11, nsteps=int(1e6))
opts.normalize_output = False  # mesolve is x3 faster if this is False


class ThreeLS_v0_env(py_environment.PyEnvironment):
    """Λ-system environment"""

    def __init__(
        self,
        Ωmax=1.0,
        Ωmin=0.0,
        Δ=0.0,
        δp=0.0,
        T=10.0,
        n_steps=20,
        γ=0.0,
        reward_gain=1.0,
        seed=1,
    ):
        """Initializes a new Λ-system environment.
        Args:
          seed: random seed for the RNG.
        """
        self.qstate = [basis(4, i) for i in range(4)]
        self.sig = [
            [self.qstate[i] * self.qstate[j].dag() for j in range(4)] for i in range(4)
        ]
        self.up = (self.sig[0][1] + self.sig[1][0]) / 2
        self.us = (self.sig[1][2] + self.sig[2][1]) / 2
        self.ψ0 = ket2dm(self.qstate[0])
        self.target_state = self.sig[2][2]

        self.Ωmax = Ωmax
        self.Ωmin = Ωmin
        self.Δ = Δ
        self.δp = δp
        self.T = T
        self.n_steps = n_steps
        self.γ = γ
        self.reward_gain = reward_gain

        self.current_step = 0
        self.current_qstate = self.ψ0
        self._state = self._dm2state(self.ψ0)
        self._episode_ended = False

        self.update()

    def update(self):
        self.H0 = self.Δ * self.sig[1][1] + self.δp * self.sig[2][2]
        self.tlist = np.linspace(0, self.T, self.n_steps + 1)
        self.Δt = self.tlist[1] - self.tlist[0]

    def action_spec(self):
        """Returns the action spec."""
        return array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            name="pulses",
            minimum=self.Ωmin,
            maximum=self.Ωmax,
        )

    def observation_spec(self):
        """Returns the observation spec."""
        return array_spec.BoundedArraySpec(
            shape=(9,),
            dtype=np.float32,
            name="density matrix",
            minimum=np.append(
                np.zeros(3, dtype=np.float32), -1 * np.ones(6, dtype=np.float32)
            ),
            maximum=np.ones(9, dtype=np.float32),
        )

    def _reset(self):
        """Resets the environment and returns the first `TimeStep` of a new episode."""
        # self._reset_next_step = False
        self.current_step = 0
        self.current_qstate = self.ψ0
        self._state = self._dm2state(self.ψ0)
        self._episode_ended = False
        return ts.restart(self._state)

    def _qstep(self, action, qstate):
        H = self.H0 + action[0] * self.up + action[1] * self.us
        L = (liouvillian(H, [np.sqrt(self.γ) * self.sig[3][1]]) * self.Δt).expm()
        return apply_superoperator(L, qstate)

    def _mesolvestep(self, action, qstate):
        H = self.H0 + action[0] * self.up + action[1] * self.us
        tlist = self.tlist[self.current_step : self.current_step + 2]
        result = mesolve(
            H, qstate, tlist, c_ops=[np.sqrt(self.γ) * self.sig[3][1]], options=opts
        )
        return result.states[-1]

    def _step(self, action):
        """Updates the environment according to the action."""

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if self.current_step < self.n_steps:
            self.current_qstate = self._qstep(action, self.current_qstate)
            # self.current_qstate = self._mesolvestep(action, self.current_qstate)
            next_state = self._dm2state(self.current_qstate)
            terminal = False
            reward = (
                0.0  # self.reward_gain * expect(self.target_state, self.current_qstate)
            )

            if self.current_step == self.n_steps - 1:
                reward = self.reward_gain * expect(
                    self.target_state, self.current_qstate
                )
                terminal = True
        else:
            terminal = True
            reward = 0
            next_state = 0
        self.current_step += 1

        if terminal:
            self._episode_ended = True
            return ts.termination(next_state, reward)
        else:
            return ts.transition(next_state, reward)

    def _dm2state(self, dm):
        return np.append(
            dm.diag()[:-1],
            np.append(
                dm.full()[([0, 0, 1], [1, 2, 2])].real,
                dm.full()[([0, 0, 1], [1, 2, 2])].imag,
            ),
        ).astype(np.float32)

    def run_evolution(self, amps):
        time_step = self.reset()
        time_step_list = [time_step]

        for i in range(self.n_steps):
            time_step = self.step(amps[i])
            time_step_list.append(time_step)

        assert time_step.is_last() == True

        state_list = np.array([x.observation for x in time_step_list])
        reward_list = np.array([x.reward for x in time_step_list])
        terminal_list = np.array([x.step_type for x in time_step_list])

        return state_list, reward_list, terminal_list

    def run_qstepevolution(self, amps):
        Ωp = amps[:, 0]
        Ωs = amps[:, 1]

        states = [self.ψ0]