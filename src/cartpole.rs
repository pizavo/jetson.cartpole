use rand::Rng;
use serde::{Deserialize, Serialize};

/// CartPole environment following the classic control problem
/// The goal is to balance a pole on a cart by moving the cart left or right
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartPoleEnv {
    // Physics constants
    pub gravity: f32,
    pub mass_cart: f32,
    pub mass_pole: f32,
    pub total_mass: f32,
    pub length: f32, // half-length of the pole
    pub pole_mass_length: f32,
    pub force_mag: f32,
    pub tau: f32, // time step

    // State variables
    pub x: f32,           // cart position
    pub x_dot: f32,       // cart velocity
    pub theta: f32,       // pole angle (radians)
    pub theta_dot: f32,   // pole angular velocity

    // Episode tracking
    pub steps: u32,
    pub max_steps: u32,

    // Thresholds for episode termination
    pub x_threshold: f32,
    pub theta_threshold_radians: f32,
}

impl Default for CartPoleEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl CartPoleEnv {
    pub fn new() -> Self {
        let gravity = 9.8;
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let total_mass = mass_cart + mass_pole;
        let length = 0.5; // half-length
        let pole_mass_length = mass_pole * length;
        let force_mag = 5.0;  // Reduced from 10.0 for gentler control
        let tau = 0.02; // 20ms per step

        let x_threshold = 2.4;
        let theta_threshold_radians = 30.0 * 2.0 * std::f32::consts::PI / 360.0;  // Increased from 12° to 30° for more recovery room

        CartPoleEnv {
            gravity,
            mass_cart,
            mass_pole,
            total_mass,
            length,
            pole_mass_length,
            force_mag,
            tau,
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            max_steps: 500,
            x_threshold,
            theta_threshold_radians,
        }
    }

    /// Reset the environment to initial state
    pub fn reset(&mut self) -> [f32; 4] {
        let mut rng = rand::rng();

        // Initialize with very small random values for human play
        // Pole starts nearly vertical (theta close to 0)
        self.x = rng.random_range(-0.01..0.01);
        self.x_dot = rng.random_range(-0.01..0.01);
        self.theta = rng.random_range(-0.02..0.02);  // ~1 degree tilt
        self.theta_dot = rng.random_range(-0.01..0.01);
        self.steps = 0;

        self.get_state()
    }

    /// Get current state as array [x, x_dot, theta, theta_dot]
    pub fn get_state(&self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }

    /// Take a step in the environment
    /// action: 0 = push left, 1 = push right, 2 = no force (natural physics)
    /// Returns: (state, reward, done)
    pub fn step(&mut self, action: i32) -> ([f32; 4], f32, bool) {
        let force = match action {
            0 => -self.force_mag,  // Push left
            1 => self.force_mag,   // Push right
            _ => 0.0,              // No force (natural physics)
        };

        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        // Physics equations from the CartPole problem
        let temp = (force + self.pole_mass_length * self.theta_dot.powi(2) * sin_theta) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp) /
            (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta.powi(2) / self.total_mass));
        let x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass;

        // Update state using Euler's method
        self.x += self.tau * self.x_dot;
        self.x_dot += self.tau * x_acc;
        self.theta += self.tau * self.theta_dot;
        self.theta_dot += self.tau * theta_acc;

        self.steps += 1;

        // Check if episode is done
        let done = self.x.abs() > self.x_threshold
            || self.theta.abs() > self.theta_threshold_radians
            || self.steps >= self.max_steps;

        // Reward is 1.0 for every step the pole is balanced
        let reward = if done { 0.0 } else { 1.0 };

        (self.get_state(), reward, done)
    }

    /// Check if the episode is done
    /// Used by Python bindings for AI training
    #[allow(dead_code)]
    pub fn is_done(&self) -> bool {
        self.x.abs() > self.x_threshold
            || self.theta.abs() > self.theta_threshold_radians
            || self.steps >= self.max_steps
    }

    /// Get the current reward
    /// Used by Python bindings for AI training
    #[allow(dead_code)]
    pub fn get_reward(&self) -> f32 {
        if self.is_done() {
            0.0
        } else {
            1.0
        }
    }

    /// Set custom parameters
    /// Used by Python bindings for AI training
    #[allow(dead_code)]
    pub fn set_params(&mut self, gravity: f32, force_mag: f32, tau: f32) {
        self.gravity = gravity;
        self.force_mag = force_mag;
        self.tau = tau;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset() {
        let mut env = CartPoleEnv::new();
        let state = env.reset();
        assert_eq!(state.len(), 4);
        assert!(state[0].abs() <= 0.05); // x
        assert!(state[2].abs() <= 0.05); // theta
    }

    #[test]
    fn test_step() {
        let mut env = CartPoleEnv::new();
        env.reset();
        let (state, reward, done) = env.step(1);
        assert_eq!(state.len(), 4);
        assert!(reward >= 0.0);
        assert!(done == true || done == false);
    }

    #[test]
    fn test_episode() {
        let mut env = CartPoleEnv::new();
        env.reset();
        let mut total_reward = 0.0;
        let mut done = false;

        while !done && env.steps < 100 {
            let (_, reward, is_done) = env.step(1);
            total_reward += reward;
            done = is_done;
        }

        assert!(total_reward >= 0.0);
    }
}

