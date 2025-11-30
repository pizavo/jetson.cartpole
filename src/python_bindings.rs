#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;
use crate::cartpole::CartPoleEnv;

/// Python wrapper for CartPole environment
#[cfg(feature = "python")]
#[pyclass]
struct PyCartPole {
    env: CartPoleEnv,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCartPole {
    #[new]
    fn new() -> Self {
        PyCartPole {
            env: CartPoleEnv::new(),
        }
    }

    /// Reset the environment
    fn reset(&mut self) -> Vec<f32> {
        self.env.reset().to_vec()
    }

    /// Take a step with given action (0=left, 1=right, 2=no force)
    /// Returns: (state, reward, done)
    fn step(&mut self, action: i32) -> (Vec<f32>, f32, bool) {
        let (state, reward, done) = self.env.step(action);
        (state.to_vec(), reward, done)
    }

    /// Get current state
    fn get_state(&self) -> Vec<f32> {
        self.env.get_state().to_vec()
    }

    /// Check if episode is done
    fn is_done(&self) -> bool {
        self.env.is_done()
    }

    /// Get current reward
    fn get_reward(&self) -> f32 {
        self.env.get_reward()
    }

    /// Get current step count
    fn get_steps(&self) -> u32 {
        self.env.steps
    }

    /// Set custom parameters
    fn set_params(&mut self, gravity: f32, force_mag: f32, tau: f32) {
        self.env.set_params(gravity, force_mag, tau);
    }

    /// Get observation space bounds
    /// Returns: (low, high) as lists
    fn observation_space(&self) -> (Vec<f32>, Vec<f32>) {
        let low = vec![-4.8, -f32::INFINITY, -0.418, -f32::INFINITY];
        let high = vec![4.8, f32::INFINITY, 0.418, f32::INFINITY];
        (low, high)
    }

    /// Get action space size
    fn action_space(&self) -> i32 {
        3  // 0=left, 1=right, 2=no force
    }
}

/// Python module
#[cfg(feature = "python")]
#[pymodule]
fn cartpole(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCartPole>()?;
    Ok(())
}

