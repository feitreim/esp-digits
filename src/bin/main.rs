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
use nalgebra::SVector;
use rust_esp_test::Model;

// This creates a default app-descriptor required by the esp-idf bootloader.
esp_bootloader_esp_idf::esp_app_desc!();

mod f {
    include!("../model_fixture.rs");
}

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

    let z = SVector::<f32, 784>::from_column_slice(&f::INPUT);

    loop {
        let start = Instant::now();
        let out = model.forward(z, f::T);
        let elapsed = start.elapsed();

        let max_err = out
            .iter()
            .zip(f::EXPECTED)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("max_err={} time={}", max_err, elapsed);

        let delay = Instant::now();
        while delay.elapsed() < Duration::from_millis(1000) {}
    }
}
