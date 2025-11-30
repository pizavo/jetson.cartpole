mod cartpole;

use cartpole::CartPoleEnv;
use ggez::event::{self, EventHandler};
use ggez::graphics::{Canvas, Color, DrawMode, Mesh, Rect, Text};
use ggez::input::keyboard::{KeyCode, KeyInput};
use ggez::{Context, GameResult};
use ggez::glam::Vec2;

const WINDOW_WIDTH: f32 = 1920.0;
const WINDOW_HEIGHT: f32 = 1080.0;
const SCALE: f32 = 100.0; // pixels per meter

struct GameState {
    env: CartPoleEnv,
    done: bool,
    total_reward: f32,
    auto_mode: bool,
    current_action: i32,  // Action to apply continuously
    left_pressed: bool,
    right_pressed: bool,
    frame_counter: u32,   // For slowing down physics updates
    failure_reason: Option<String>,  // Why the episode ended
}

impl GameState {
    fn new() -> Self {
        let mut env = CartPoleEnv::new();
        env.reset();

        GameState {
            env,
            done: false,
            total_reward: 0.0,
            auto_mode: false,
            current_action: 2,  // No force - natural physics
            left_pressed: false,
            right_pressed: false,
            frame_counter: 0,
            failure_reason: None,
        }
    }

    fn reset(&mut self) {
        self.env.reset();
        self.done = false;
        self.total_reward = 0.0;
        self.current_action = 2;  // No force
        self.left_pressed = false;
        self.right_pressed = false;
        self.frame_counter = 0;
        self.failure_reason = None;
    }

    fn take_action(&mut self, action: i32) {
        if !self.done {
            let (_, reward, done) = self.env.step(action);
            self.total_reward += reward;

            if done && self.failure_reason.is_none() {
                // Determine why the episode ended
                if self.env.steps >= self.env.max_steps {
                    self.failure_reason = Some(format!("✓ MAX STEPS REACHED ({})", self.env.max_steps));
                } else if self.env.x.abs() > self.env.x_threshold {
                    let direction = if self.env.x < 0.0 { "LEFT" } else { "RIGHT" };
                    self.failure_reason = Some(format!(
                        "✗ CART OFF TRACK - {} edge ({:.2}m > {:.1}m)",
                        direction, self.env.x.abs(), self.env.x_threshold
                    ));
                } else if self.env.theta.abs() > self.env.theta_threshold_radians {
                    let direction = if self.env.theta < 0.0 { "LEFT" } else { "RIGHT" };
                    self.failure_reason = Some(format!(
                        "✗ POLE FELL {} ({:.1}° > {:.0}°)",
                        direction,
                        self.env.theta.to_degrees().abs(),
                        self.env.theta_threshold_radians.to_degrees()
                    ));
                }
            }

            self.done = done;
        }
    }

    // Simple AI: move cart towards the direction the pole is falling
    fn auto_action(&self) -> i32 {
        // If pole is tilting right (positive theta), push right
        // If pole is tilting left (negative theta), push left
        if self.env.theta > 0.0 {
            1
        } else {
            0
        }
    }

    fn update_action(&mut self) {
        // Determine current action based on key states
        if self.left_pressed && !self.right_pressed {
            self.current_action = 0;  // Push left
        } else if self.right_pressed && !self.left_pressed {
            self.current_action = 1;  // Push right
        } else {
            // No keys pressed or both pressed: apply no force (natural physics)
            self.current_action = 2;  // No force - let gravity do its thing!
        }
    }
}

impl EventHandler for GameState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        // Update physics every 2 frames for more human-playable speed (30 updates/sec instead of 60)
        self.frame_counter += 1;
        if self.frame_counter >= 2 {
            self.frame_counter = 0;

            // Always update physics, even when no keys are pressed
            if !self.done {
                let action = if self.auto_mode {
                    self.auto_action()
                } else {
                    self.current_action
                };
                self.take_action(action);
            }
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, Color::from_rgb(0, 0, 0));

        // Calculate positions
        let center_x = WINDOW_WIDTH / 2.0;
        let ground_y = WINDOW_HEIGHT / 2.0 + 100.0;

        // Draw ground line
        let ground_line = Mesh::new_line(
            ctx,
            &[
                Vec2::new(0.0, ground_y),
                Vec2::new(WINDOW_WIDTH, ground_y),
            ],
            2.0,
            Color::from_rgb(100, 100, 100),
        )?;
        canvas.draw(&ground_line, Vec2::ZERO);

        // Draw cart position marker at center
        let center_marker = Mesh::new_line(
            ctx,
            &[
                Vec2::new(center_x, ground_y + 5.0),
                Vec2::new(center_x, ground_y + 15.0),
            ],
            2.0,
            Color::from_rgb(200, 100, 100),
        )?;
        canvas.draw(&center_marker, Vec2::ZERO);

        // Draw position boundary markers (left and right limits)
        let left_boundary_x = center_x - self.env.x_threshold * SCALE;
        let right_boundary_x = center_x + self.env.x_threshold * SCALE;

        // Left boundary
        let left_boundary = Mesh::new_line(
            ctx,
            &[
                Vec2::new(left_boundary_x, ground_y - 20.0),
                Vec2::new(left_boundary_x, ground_y + 20.0),
            ],
            3.0,
            Color::from_rgba(255, 100, 100, 150),  // Semi-transparent red
        )?;
        canvas.draw(&left_boundary, Vec2::ZERO);

        // Right boundary
        let right_boundary = Mesh::new_line(
            ctx,
            &[
                Vec2::new(right_boundary_x, ground_y - 20.0),
                Vec2::new(right_boundary_x, ground_y + 20.0),
            ],
            3.0,
            Color::from_rgba(255, 100, 100, 150),  // Semi-transparent red
        )?;
        canvas.draw(&right_boundary, Vec2::ZERO);

        // Cart position in screen coordinates
        let cart_x = center_x + self.env.x * SCALE;
        let cart_y = ground_y;

        // Draw cart (rectangle)
        let cart_width = 50.0;
        let cart_height = 30.0;
        let cart_rect = Mesh::new_rectangle(
            ctx,
            DrawMode::fill(),
            Rect::new(
                cart_x - cart_width / 2.0,
                cart_y - cart_height,
                cart_width,
                cart_height,
            ),
            Color::from_rgb(50, 100, 200),
        )?;
        canvas.draw(&cart_rect, Vec2::ZERO);

        // Draw cart outline
        let cart_outline = Mesh::new_rectangle(
            ctx,
            DrawMode::stroke(2.0),
            Rect::new(
                cart_x - cart_width / 2.0,
                cart_y - cart_height,
                cart_width,
                cart_height,
            ),
            Color::from_rgb(30, 60, 120),
        )?;
        canvas.draw(&cart_outline, Vec2::ZERO);

        // Draw pole
        let pole_length = self.env.length * 2.0 * SCALE; // full length in pixels
        let pole_end_x = cart_x + pole_length * self.env.theta.sin();
        let pole_end_y = cart_y - cart_height / 2.0 - pole_length * self.env.theta.cos();

        let pole_line = Mesh::new_line(
            ctx,
            &[
                Vec2::new(cart_x, cart_y - cart_height / 2.0),
                Vec2::new(pole_end_x, pole_end_y),
            ],
            8.0,
            Color::from_rgb(200, 50, 50),
        )?;
        canvas.draw(&pole_line, Vec2::ZERO);

        // Draw pole end (mass)
        let pole_mass = Mesh::new_circle(
            ctx,
            DrawMode::fill(),
            Vec2::new(pole_end_x, pole_end_y),
            12.0,
            0.1,
            Color::from_rgb(150, 30, 30),
        )?;
        canvas.draw(&pole_mass, Vec2::ZERO);

        // Draw angle threshold indicators (faint lines showing safe zone)
        let threshold_angle = self.env.theta_threshold_radians;
        let threshold_length = pole_length * 0.9;  // Slightly shorter than pole

        // Left threshold line
        let left_threshold_x = cart_x - threshold_length * threshold_angle.sin();
        let left_threshold_y = cart_y - cart_height / 2.0 - threshold_length * threshold_angle.cos();
        let left_threshold = Mesh::new_line(
            ctx,
            &[
                Vec2::new(cart_x, cart_y - cart_height / 2.0),
                Vec2::new(left_threshold_x, left_threshold_y),
            ],
            2.0,
            Color::from_rgba(255, 100, 100, 100),  // Semi-transparent red
        )?;
        canvas.draw(&left_threshold, Vec2::ZERO);

        // Right threshold line
        let right_threshold_x = cart_x + threshold_length * threshold_angle.sin();
        let right_threshold_y = cart_y - cart_height / 2.0 - threshold_length * threshold_angle.cos();
        let right_threshold = Mesh::new_line(
            ctx,
            &[
                Vec2::new(cart_x, cart_y - cart_height / 2.0),
                Vec2::new(right_threshold_x, right_threshold_y),
            ],
            2.0,
            Color::from_rgba(255, 100, 100, 100),  // Semi-transparent red
        )?;
        canvas.draw(&right_threshold, Vec2::ZERO);

        // Draw force indicator (only when force is being applied)
        if !self.done && self.current_action != 2 {
            let force_length = 30.0;
            let force_x = if self.current_action == 1 {
                cart_x + cart_width / 2.0 + force_length
            } else {
                cart_x - cart_width / 2.0 - force_length
            };

            let force_arrow = Mesh::new_line(
                ctx,
                &[
                    Vec2::new(
                        if self.current_action == 1 { cart_x + cart_width / 2.0 } else { cart_x - cart_width / 2.0 },
                        cart_y - cart_height / 2.0,
                    ),
                    Vec2::new(force_x, cart_y - cart_height / 2.0),
                ],
                3.0,
                Color::from_rgb(255, 200, 0),
            )?;
            canvas.draw(&force_arrow, Vec2::ZERO);
        }

        // Draw info text
        let mut y_offset = 20.0;
        let texts = vec![
            "CartPole Game".to_string(),
            format!("Steps: {} / {}", self.env.steps, self.env.max_steps),
            format!("Reward: {:.1}", self.total_reward),
            format!("Position: {:.2}m / ±{:.1}m", self.env.x, self.env.x_threshold),
            format!("Angle: {:.2}° / ±{:.0}°",
                self.env.theta.to_degrees(),
                self.env.theta_threshold_radians.to_degrees()),
            format!("Action: {}", match self.current_action {
                0 => "← LEFT",
                1 => "RIGHT →",
                _ => "NO FORCE (gravity only)",
            }),
            "".to_string(),
            "Controls:".to_string(),
            "  LEFT/RIGHT: Push cart".to_string(),
            format!("  SPACE: Auto mode ({})", if self.auto_mode { "ON" } else { "OFF" }),
            "  R: Reset".to_string(),
            "  ESC: Quit".to_string(),
            "".to_string(),
            "Pole falls naturally with gravity!".to_string(),
        ];

        if self.done {
            texts.iter().chain(std::iter::once(&"".to_string()))
                .chain(std::iter::once(&"=== EPISODE DONE ===".to_string()))
                .for_each(|_| {});
        }

        for text_str in &texts {
            let text = Text::new(text_str);
            canvas.draw(
                &text,
                Vec2::new(10.0, y_offset),
            );
            y_offset += 20.0;
        }

        if self.done {
            y_offset += 10.0;  // Extra spacing
            let done_text = Text::new("=== EPISODE DONE ===");
            canvas.draw(
                &done_text,
                Vec2::new(10.0, y_offset),
            );
            y_offset += 25.0;

            // Show the specific failure reason
            if let Some(reason) = &self.failure_reason {
                let reason_text = Text::new(reason);
                canvas.draw(
                    &reason_text,
                    Vec2::new(10.0, y_offset),
                );
                y_offset += 25.0;

                let hint_text = Text::new("Press R to reset");
                canvas.draw(
                    &hint_text,
                    Vec2::new(10.0, y_offset),
                );
            }
        }

        canvas.finish(ctx)?;
        Ok(())
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        input: KeyInput,
        repeated: bool,
    ) -> GameResult {
        // Ignore repeated key events (when key is held down)
        if repeated {
            return Ok(());
        }

        if let Some(keycode) = input.keycode {
            match keycode {
                KeyCode::Left => {
                    self.left_pressed = true;
                    self.update_action();
                }
                KeyCode::Right => {
                    self.right_pressed = true;
                    self.update_action();
                }
                KeyCode::R => self.reset(),
                KeyCode::Space => self.auto_mode = !self.auto_mode,
                KeyCode::Escape => std::process::exit(0),
                _ => {}
            }
        }
        Ok(())
    }

    fn key_up_event(
        &mut self,
        _ctx: &mut Context,
        input: KeyInput,
    ) -> GameResult {
        if let Some(keycode) = input.keycode {
            match keycode {
                KeyCode::Left => {
                    self.left_pressed = false;
                    self.update_action();
                }
                KeyCode::Right => {
                    self.right_pressed = false;
                    self.update_action();
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn main() -> GameResult {
    println!("Starting CartPole Game...");
    println!("This Rust implementation can be used with Python for AI training!");
    println!();

    let cb = ggez::ContextBuilder::new("cartpole", "AI Training")
        .window_setup(ggez::conf::WindowSetup::default().title("CartPole - Balance the Pole!"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(WINDOW_WIDTH, WINDOW_HEIGHT));

    let (ctx, event_loop) = cb.build()?;
    let state = GameState::new();

    event::run(ctx, event_loop, state)
}