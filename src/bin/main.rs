#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};

use esp_println::println;
use heapless::String;
use nalgebra::SVector;
use rust_esp_test::{Model, Rng};

// This creates a default app-descriptor required by the esp-idf bootloader.
esp_bootloader_esp_idf::esp_app_desc!();

mod f {
    include!("../model_fixture.rs");
}

// Each shade glyph is a 3-byte UTF-8 char, plus one '\n' per row.
const CANVAS_SIZE: usize = (28 * 28 * 3) + 28;

fn pixel_to_char(pix: f32) -> char {
    const SHADES: [char; 5] = [' ', '░', '▒', '▓', '█'];
    let i = (pix * SHADES.len() as f32) as usize;
    SHADES[i.min(SHADES.len() - 1)]
}

fn print_digit(pixels: SVector<f32, 784>) -> String<CANVAS_SIZE> {
    let mut s = String::new();
    for y in 0..28 {
        for x in 0..28 {
            let pix = pixels[(y * 28 + x) as usize];
            s.push(pixel_to_char(pix)).ok();
        }
        s.push('\n').ok();
    }
    s
}

const N: usize = 1000;

#[allow(
    clippy::large_stack_frames,
    reason = "the denoiser threads a handful of 784- and 256-wide vectors through the stack"
)]
#[main]
fn main() -> ! {
    esp_println::logger::init_logger_from_env();

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    let model = Model::new(
        &f::FREQS,
        &f::TIME_W1, &f::TIME_B1, &f::TIME_W2, &f::TIME_B2,
        &f::INP_W, &f::INP_B,
        [&f::NORM_G0, &f::NORM_G1, &f::NORM_G2],
        [&f::NORM_B0, &f::NORM_B1, &f::NORM_B2],
        [&f::FC_W0, &f::FC_W1, &f::FC_W2],
        [&f::FC_B0, &f::FC_B1, &f::FC_B2],
        &f::OUT_NORM_G, &f::OUT_NORM_B,
        &f::OUT_W, &f::OUT_B,
    );

    let mut rng = Rng::new(0xC0FFEE);

    loop {
        let mut z = rng.randn::<784>(); // t=0: fresh Gaussian noise
        let start = Instant::now();
        let dt = 1.0 / N as f32;
        let mut t = 0.0;
        for _ in 0..N-1 {
            let x_hat = model.forward(z,t);
            z += (x_hat - z) / (1.0 - t) * dt;
            t += dt;
        }
        let elapsed = start.elapsed().as_millis() as f32 / 1_000.0;

        // Model works in (-1, 1) remap to (0, 1) for the shade ramp.
        let image = z.map(|p| (p.clamp(-1.0, 1.0) + 1.0) / 2.0);
        println!("{}", print_digit(image));
        println!("sampled {} steps in {:.3} seconds", N, elapsed);

        let delay = Instant::now();
        while delay.elapsed() < Duration::from_millis(1000) {}
    }
}
