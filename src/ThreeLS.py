
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