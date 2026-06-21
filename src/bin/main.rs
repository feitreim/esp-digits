#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};

use esp_println::println;
use heapless::String;
use nalgebra::{Const, MatrixView, SVector};

// This creates a default app-descriptor required by the esp-idf bootloader.
// For more information see: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/app_image_format.html#application-description>
esp_bootloader_esp_idf::esp_app_desc!();

mod fixture {
    include!("../fixture.rs");
}

// utilitys for printing outputs on the esp32
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

fn mlp(
    input: SVector<f32, 784>,
    up_proj: MatrixView<f32, Const<256>, Const<784>>,
    bias: SVector<f32, 256>,
    down_proj: MatrixView<f32, Const<10>, Const<256>>,
) -> SVector<f32, 10> {
    let z = up_proj * input + bias;
    let h = z.map(|x| x.max(0.0));
    down_proj * h
}

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[main]
fn main() -> ! {
    // generator version: 1.3.0
    // generator parameters: --chip esp32 -o log -o esp-backtrace -o zed

    esp_println::logger::init_logger_from_env();

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    let up_proj = MatrixView::<f32, Const<256>, Const<784>>::from_slice(&fixture::UP_PROJ);
    let down_proj = MatrixView::<f32, Const<10>, Const<256>>::from_slice(&fixture::DOWN_PROJ);
    let bias = SVector::<f32, 256>::from_column_slice(&fixture::BIAS);
    let input = SVector::<f32, 784>::from_column_slice(&fixture::INPUT);

    loop {
        println!("{}", print_digit(input));
        let computation_start = Instant::now();
        let logits = mlp(input, up_proj, bias, down_proj);
        let computation_time = computation_start.elapsed();

        let pred = logits.argmax().0;

        let max_err = logits
            .iter()
            .zip(fixture::EXPECTED_LOGITS)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        esp_println::println!(
            "pred={} truth={} match={} max_logit_err={}, time_elapsed={}",
            pred,
            fixture::LABEL,
            pred == fixture::LABEL,
            max_err,
            computation_time
        );

        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(500) {}
    }
}
