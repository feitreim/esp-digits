#![no_std]
use nalgebra::{ComplexField, Const, MatrixView, SVector, SVectorView};

const EPS: f32 = 1e-5;

/// xorshift64* PRNG with a Box–Muller transform for standard-normal samples.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1) // a zero state is a fixed point of xorshift
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn uniform(&mut self) -> f32 {
        // top 24 bits → [0, 1), the mantissa width of f32
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    pub fn randn<const N: usize>(&mut self) -> SVector<f32, N> {
        SVector::from_fn(|_, _| {
            let u1 = self.uniform().max(1e-7); // keep ln finite
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
        })
    }
}

#[derive(Debug)]
struct Linear<'a, const IN: usize, const OUT: usize> {
    w: MatrixView<'a, f32, Const<OUT>, Const<IN>>,
    b: Option<SVectorView<'a, f32, OUT>>,
}

impl<'a, const IN: usize, const OUT: usize> Linear<'a, IN, OUT> {
    fn new(
        w: MatrixView<'a, f32, Const<OUT>, Const<IN>>,
        b: Option<SVectorView<'a, f32, OUT>>,
    ) -> Self {
        Self { w, b }
    }

    fn linear(&self, x: SVector<f32, IN>) -> SVector<f32, OUT> {
        match self.b {
            Some(b) => self.w * x + b,
            None => self.w * x,
        }
    }
}

#[derive(Debug)]
struct LayerNorm<'a, const DIM: usize> {
    gamma: SVectorView<'a, f32, DIM>,
    beta: SVectorView<'a, f32, DIM>,
}

impl<'a, const DIM: usize> LayerNorm<'a, DIM> {
    fn layer_norm(&self, mut x: SVector<f32, DIM>) -> SVector<f32, DIM> {
        let mean = x.mean();
        let std = (x.variance() + EPS).sqrt();
        x.add_scalar_mut(-mean);
        x.unscale_mut(std);
        x.component_mul_assign(&self.gamma);
        x += self.beta;
        x
    }
}

fn silu<const DIM: usize>(mut x: SVector<f32, DIM>) -> SVector<f32, DIM> {
    x.apply(|a| *a *= 1.0 / (1.0 + (-*a).exp()));
    x
}

#[derive(Debug)]
struct TimestepEmbedding<'a, const HALF: usize> {
    freqs: SVectorView<'a, f32, HALF>,
}

impl<'a, const HALF: usize> TimestepEmbedding<'a, HALF> {
    // t should be continuous (0, 1)
    fn timestep_embedding<const DIM: usize>(&self, t: f32) -> SVector<f32, DIM> {
        debug_assert_eq!(DIM, 2 * HALF);
        let scaled = self.freqs.scale(t * 1000f32);
        let mut out = SVector::<f32, DIM>::zeros();
        out.rows_mut(0, HALF).copy_from(&scaled.map(f32::sin));
        out.rows_mut(HALF, HALF).copy_from(&scaled.map(f32::cos));
        out
    }
}

struct TimeBlock<'a, const EMB: usize, const DIM: usize, const HALF: usize> {
    embedding: TimestepEmbedding<'a, HALF>,
    linear1: Linear<'a, EMB, DIM>,
    linear2: Linear<'a, DIM, DIM>,
}

impl<'a, const EMB: usize, const DIM: usize, const HALF: usize> TimeBlock<'a, EMB, DIM, HALF> {
    fn time_block(&self, t: f32) -> SVector<f32, DIM> {
        let embed = self.embedding.timestep_embedding::<EMB>(t);
        let h = self.linear1.linear(embed);
        self.linear2.linear(silu(h))
    }
}

#[derive(Debug)]
struct AddBlock<'a, const DIM: usize> {
    norm: LayerNorm<'a, DIM>,
    fc: Linear<'a, DIM, DIM>,
}

impl<'a, const DIM: usize> AddBlock<'a, DIM> {
    fn add_block(&self, h: SVector<f32, DIM>, temb: SVector<f32, DIM>) -> SVector<f32, DIM> {
        h + self.fc.linear(silu(self.norm.layer_norm(h) + temb))
    }
}

const IMG: usize = 784;
const HIDDEN: usize = 256;
const EMB: usize = 128;
const HALF: usize = EMB / 2;
const BLOCKS: usize = 3;

pub struct Model<'a> {
    time: TimeBlock<'a, EMB, HIDDEN, HALF>,
    inp: Linear<'a, IMG, HIDDEN>,
    blocks: [AddBlock<'a, HIDDEN>; BLOCKS],
    out_norm: LayerNorm<'a, HIDDEN>,
    out: Linear<'a, HIDDEN, IMG>,
}

fn lin<'a, const IN: usize, const OUT: usize>(w: &'a [f32], b: &'a [f32]) -> Linear<'a, IN, OUT> {
    let w = MatrixView::<f32, Const<OUT>, Const<IN>>::from_slice(w);
    Linear::new(w, Some(SVectorView::<f32, OUT>::from_slice(b)))
}

fn norm<'a, const DIM: usize>(gamma: &'a [f32], beta: &'a [f32]) -> LayerNorm<'a, DIM> {
    LayerNorm {
        gamma: SVectorView::<f32, DIM>::from_slice(gamma),
        beta: SVectorView::<f32, DIM>::from_slice(beta),
    }
}

impl<'a> Model<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        freqs: &'a [f32],
        time_w1: &'a [f32],
        time_b1: &'a [f32],
        time_w2: &'a [f32],
        time_b2: &'a [f32],
        inp_w: &'a [f32],
        inp_b: &'a [f32],
        norm_g: [&'a [f32]; BLOCKS],
        norm_b: [&'a [f32]; BLOCKS],
        fc_w: [&'a [f32]; BLOCKS],
        fc_b: [&'a [f32]; BLOCKS],
        out_norm_g: &'a [f32],
        out_norm_b: &'a [f32],
        out_w: &'a [f32],
        out_b: &'a [f32],
    ) -> Self {
        Self {
            time: TimeBlock {
                embedding: TimestepEmbedding {
                    freqs: SVectorView::from_slice(freqs),
                },
                linear1: lin(time_w1, time_b1),
                linear2: lin(time_w2, time_b2),
            },
            inp: lin(inp_w, inp_b),
            blocks: core::array::from_fn(|i| AddBlock {
                norm: norm(norm_g[i], norm_b[i]),
                fc: lin(fc_w[i], fc_b[i]),
            }),
            out_norm: norm(out_norm_g, out_norm_b),
            out: lin(out_w, out_b),
        }
    }

    pub fn forward(&self, z: SVector<f32, IMG>, t: f32) -> SVector<f32, IMG> {
        let temb = self.time.time_block(t);
        let mut h = self.inp.linear(z);
        for block in &self.blocks {
            h = block.add_block(h, temb);
        }
        self.out.linear(self.out_norm.layer_norm(h))
    }
}
