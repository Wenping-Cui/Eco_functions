"""
Test suite for the Eco_function package.

Run with:
    python -m pytest test/
or directly:
    python test/test_eco_functions.py
"""
import sys
import os
import unittest
import numpy as np

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Eco_function.eco_lib import (
    Ecology_simulation,
    Consum_matrix_MA,
    Entropy_cal,
    Make_consumption_matrices,
)
from Eco_function.eco_func import KL, MP_density, KL_lams, appendDFToCSV_void
from Eco_function.C_matrix import Consum_matrix, block, circ, binomial
from Eco_function.D_matrix import Make_D_matrices
from Eco_function.Model_cavity_CRM import CRM_Cavity_simulation
from Eco_function.Model_cavity_LV import LV_Cavity_simulation, LVModel


# ---------------------------------------------------------------------------
# Helper: minimal CRM parameter dict
# ---------------------------------------------------------------------------

def _crm_params(S=10, M=10, sample_size=3):
    return {
        'sample_size': sample_size,
        'S': S,
        'M': M,
        'K': 1.0,
        'sigma_K': 0.1,
        'mu': 0.5,
        'sigma_c': 1.0,
        'm': 0.1,
        'sigma_m': 0.01,
        'loop_size': 5,
        't0': 0,
        't1': 100,
        'Nt': 200,
    }


def _lv_params(S=10, sample_size=3):
    return {
        'sample_size': sample_size,
        'S': S,
        'k': 1.0,
        'sigma_k': 0.1,
        'g': 1.0,
        'sigma_g': 0.0,
        'mu': 0.0,
        'epsilon': 0.1,
        'B': 0,
        'ODE_Time': [0, 100, 200],
        'mapping_CR': False,
        'sys_gamma': 0,
        'A_type': 'gaussian',
        'B_type': 'null',
    }


# ===========================================================================
class TestCMatrix(unittest.TestCase):
    """Tests for C_matrix.py."""

    def test_consum_matrix_shape(self):
        c = Consum_matrix(c_m=1.0, c_sigma=0.1, q_c=0.5, M=20, F=4, flag_family=0)
        self.assertEqual(c.shape, (20,))

    def test_consum_matrix_normalisation(self):
        c = Consum_matrix(c_m=1.0, c_sigma=0.01, q_c=0.5, M=20, F=4, flag_family=0)
        # Sum should be close to M * c_m (= 20) up to rescaling of c[0]
        self.assertGreater(np.sum(c), 0)

    def test_block_shape(self):
        C = block(n=3, m=4)
        self.assertEqual(C.shape, (12, 12))

    def test_block_values(self):
        C = block(n=2, m=3)
        # Diagonal blocks should be all-ones; off-diagonal blocks zero
        self.assertTrue(np.all(C[:3, :3] == 1))
        self.assertTrue(np.all(C[:3, 3:] == 0))

    def test_circ_shape(self):
        C = circ(n=10, r=2)
        self.assertEqual(C.shape, (10, 10))

    def test_circ_symmetry(self):
        C = circ(n=8, r=1.5)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_binomial_shape(self):
        C = binomial(p=0.3, S=5, M=8)
        self.assertEqual(C.shape, (5, 8))

    def test_binomial_values(self):
        C = binomial(p=0.3, S=20, M=20)
        self.assertTrue(np.all((C == 0) | (C == 1)))


# ===========================================================================
class TestDMatrix(unittest.TestCase):
    """Tests for D_matrix.py."""

    def _energies(self, M=10):
        return np.ones(M)

    def test_make_D_fixed_length(self):
        eco = Make_D_matrices(S=5, M=10, nu=2, p=0.1, q=0.8,
                              energies=self._energies(10), flag=('fixed', ''))
        self.assertEqual(len(eco), 5)
        self.assertEqual(eco[0].shape, (10, 10))

    def test_make_D_random_length(self):
        eco = Make_D_matrices(S=5, M=10, nu=2, p=0.1, q=0.8,
                              energies=self._energies(10), flag=('random', ''))
        self.assertEqual(len(eco), 5)

    def test_make_D_one_step(self):
        eco = Make_D_matrices(S=5, M=8, nu=1, p=0.2, q=1.0,
                              energies=self._energies(8), flag=('one-step', ''))
        self.assertEqual(len(eco), 7)  # M - 1

    def test_make_D_trophic_ordering(self):
        """D[a, b] should be zero whenever a <= b (no backward flow)."""
        eco = Make_D_matrices(S=3, M=10, nu=2, p=0.3, q=1.0,
                              energies=self._energies(10), flag=('fixed', ''))
        for D in eco:
            for a in range(10):
                for b in range(a, 10):
                    self.assertEqual(D[a, b], 0.0,
                                     msg=f"D[{a},{b}]={D[a,b]} should be 0")


# ===========================================================================
class TestEcoLib(unittest.TestCase):
    """Tests for eco_lib.py utility functions."""

    def test_consum_matrix_MA_shape(self):
        C = Consum_matrix_MA(p=0.3, S=10, M=15)
        self.assertEqual(C.shape, (10, 15))

    def test_consum_matrix_MA_binary(self):
        C = Consum_matrix_MA(p=0.5, S=20, M=20)
        self.assertTrue(np.all((C == 0) | (C == 1)))

    def test_entropy_cal_positive(self):
        N = np.array([1.0, 2.0, 3.0, 4.0])
        H = Entropy_cal(N)
        self.assertGreater(H, 0)

    def test_entropy_cal_uniform(self):
        """Uniform distribution should have maximum entropy = log(n)."""
        N = np.ones(8)
        H = Entropy_cal(N)
        np.testing.assert_allclose(H, np.log(8), rtol=1e-10)

    def test_make_consumption_matrices_shape(self):
        eco = Make_consumption_matrices(S=5, M=10, nu=3, p=0.2, q=0.5,
                                        metabolic_flag='fixed')
        self.assertEqual(len(eco), 5)
        self.assertEqual(eco[0].shape, (10, 10))

    def test_make_consumption_matrices_one_step(self):
        eco = Make_consumption_matrices(S=5, M=8, nu=1, p=0.2, q=1.0,
                                        metabolic_flag='one-step')
        self.assertEqual(len(eco), 7)  # M - 1


# ===========================================================================
class TestEcoLibSimulation(unittest.TestCase):
    """Integration tests for Ecology_simulation."""

    def _build_sim(self, flag_crossfeeding=False):
        S, M = 5, 5
        np.random.seed(42)
        C = np.abs(np.random.normal(0.5, 0.1, (S, M)))
        K = np.ones(M)
        energies = np.ones(M)
        tau_inv = np.ones(M)
        costs = 0.1 * np.ones(S)
        growth = np.ones(S)
        R_ini = 0.1 * np.ones(M)
        N_ini = 0.1 * np.ones(S)
        T_par = [0, 50, 100]
        if flag_crossfeeding:
            D = np.zeros((M, M))
            D[2, 1] = 0.1
            par = [True, M, S, R_ini, N_ini, T_par, C, energies, tau_inv,
                   costs, growth, D, range(M), K]
        else:
            par = [False, M, S, R_ini, N_ini, T_par, C, energies, tau_inv,
                   costs, growth, K]
        return Ecology_simulation(par)

    def test_linear_dynamics(self):
        sim = self._build_sim()
        sim.flag_renew = True
        sim.flag_linear = True
        R, N = sim.simulation()
        self.assertEqual(R.shape, (100, 5))
        self.assertEqual(N.shape, (100, 5))
        self.assertTrue(np.all(sim.R_f >= 0))
        self.assertTrue(np.all(sim.N_f >= 0))

    def test_nonrenew_dynamics(self):
        sim = self._build_sim()
        sim.flag_renew = False
        R, N = sim.simulation()
        self.assertTrue(np.all(sim.R_f >= 0))

    def test_survive_count(self):
        sim = self._build_sim()
        sim.flag_renew = True
        sim.flag_linear = True
        sim.simulation()
        self.assertGreaterEqual(sim.survive, 0)
        self.assertLessEqual(sim.survive, 5)

    def test_crossfeeding_dynamics(self):
        sim = self._build_sim(flag_crossfeeding=True)
        sim.e = 0.6
        R, N = sim.simulation()
        self.assertTrue(np.all(sim.R_f >= 0))


# ===========================================================================
class TestEcoFunc(unittest.TestCase):
    """Tests for eco_func.py."""

    def test_kl_zero(self):
        """KL(P, P) should be approximately zero."""
        P = np.array([0.1, 0.2, 0.3, 0.4])
        kl = KL(P, P)
        self.assertAlmostEqual(kl, 0.0, places=10)

    def test_kl_positive(self):
        P = np.array([0.1, 0.2, 0.3, 0.4])
        Q = np.array([0.4, 0.3, 0.2, 0.1])
        kl = KL(P, Q)
        self.assertGreater(kl, 0)

    def test_mp_density_shape(self):
        bins = np.linspace(0.01, 4.0, 50)
        density = MP_density(bins, a=0.01, b=4.0, c=0.5, s=1.0)
        self.assertEqual(density.shape, bins.shape)

    def test_appendDFToCSV(self):
        import pandas as pd
        import tempfile
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        # Use a path that does NOT exist yet so the header is written on first call
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            appendDFToCSV_void(df, path)             # creates with header
            appendDFToCSV_void(df, path)             # appends without header
            loaded = pd.read_csv(path, index_col=0)
            self.assertEqual(len(loaded), 4)


# ===========================================================================
class TestLVModel(unittest.TestCase):
    """Tests for the LVModel ODE solver."""

    def test_simulation_shape(self):
        np.random.seed(0)
        S = 5
        A = np.abs(np.random.normal(0.1, 0.01, (S, S)))
        par = {
            'S': S,
            'A': A,
            'k': np.ones(S),
            'g': np.ones(S),
            'ODE_Time': [0, 50, 100],
            'N_ini': 0.1 * np.ones(S),
        }
        model = LVModel(par)
        T, N = model._simulation()
        self.assertEqual(N.shape, (100, S))
        self.assertTrue(np.all(model.N_f >= 0))

    def test_survive_count(self):
        np.random.seed(1)
        S = 5
        A = np.abs(np.random.normal(0.1, 0.01, (S, S)))
        par = {
            'S': S,
            'A': A,
            'k': np.ones(S),
            'g': np.ones(S),
            'ODE_Time': [0, 100, 200],
            'N_ini': 0.1 * np.ones(S),
        }
        model = LVModel(par)
        model._simulation()
        self.assertGreaterEqual(model.survive, 0)
        self.assertLessEqual(model.survive, S)


# ===========================================================================
class TestLVCavitySimulation(unittest.TestCase):
    """Smoke tests for LV_Cavity_simulation."""

    def test_ode_simulation_returns_dict(self):
        np.random.seed(7)
        params = _lv_params(S=8, sample_size=2)
        model = LV_Cavity_simulation(params)
        result = model._simulation(dynamics='ODE')
        self.assertIn('phi_N', result)
        self.assertIn('mean_N', result)
        self.assertIn('lam_min', result)

    def test_phi_N_range(self):
        np.random.seed(8)
        params = _lv_params(S=8, sample_size=3)
        model = LV_Cavity_simulation(params)
        result = model._simulation(dynamics='ODE')
        self.assertGreaterEqual(result['phi_N'], 0.0)
        self.assertLessEqual(result['phi_N'], 1.0)


# ===========================================================================
class TestCRMCavitySimulation(unittest.TestCase):
    """Smoke tests for CRM_Cavity_simulation."""

    def test_ode_simulation_quadratic(self):
        np.random.seed(42)
        params = _crm_params(S=8, M=8, sample_size=2)
        model = CRM_Cavity_simulation(params)
        model.gamma_flag = 'S/M'
        model.B_type = 'null'
        result = model.ode_simulation(Dynamics='quadratic', Simulation_type='CVXOPT')
        self.assertIn('phi_N', result)
        self.assertIn('phi_R', result)
        self.assertGreaterEqual(result['phi_N'], 0.0)
        self.assertLessEqual(result['phi_N'], 1.0)

    def test_ode_simulation_linear(self):
        np.random.seed(99)
        params = _crm_params(S=6, M=6, sample_size=2)
        model = CRM_Cavity_simulation(params)
        model.gamma_flag = 'S/M'
        model.B_type = 'null'
        result = model.ode_simulation(Dynamics='linear', Simulation_type='CVXOPT')
        self.assertIn('chi', result)
        self.assertIn('nu', result)

    def test_initialize_random_variable(self):
        np.random.seed(5)
        params = _crm_params(S=10, M=10)
        model = CRM_Cavity_simulation(params)
        sim_pars = model.initialize_random_variable()
        self.assertEqual(len(sim_pars), 12)  # non-crossfeeding list length


# ===========================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
