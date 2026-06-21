#![no_std]
use nalgebra::{ComplexField, Const, MatrixView, SMatrix, SVector, SVectorView};

#[derive(Debug)]
struct Linear<'a, const OUT: usize, const IN: usize> {
    w: MatrixView<'a, f32, Const<OUT>, Const<IN>>,
    b: Option<SVectorView<'a, f32, OUT>>,
}

impl<'a, const OUT: usize, const IN: usize> Linear<'a, OUT, IN> {
    fn new(
        w: MatrixView<'a, f32, Const<OUT>, Const<IN>>,
        b: Option<SVectorView<'a, f32, OUT>>,
    ) -> Self {
        Self { w, b }
    }
}

fn linear<'a, const OUT: usize, const IN: usize, const SEQ: usize>(
    x: SMatrix<f32, IN, SEQ>,
    linear: Linear<'a, OUT, IN>,
) -> SMatrix<f32, OUT, SEQ> {
    let mut y = linear.w * x;
    if let Some(b) = linear.b {
        y.column_iter_mut().for_each(|mut c| c += b);
    };
    y
}

#[derive(Debug)]
struct LayerNorm<'a, const DIM: usize> {
    gamma: SVectorView<'a, f32, DIM>,
    beta: SVectorView<'a, f32, DIM>,
}

const EPS: f32 = 1e-5;

fn layer_norm<'a, const DIM: usize, const SEQ: usize>(
    mut x: SMatrix<f32, DIM, SEQ>,
    norm: LayerNorm<'a, DIM>,
) -> SMatrix<f32, DIM, SEQ> {
    x.column_iter_mut().for_each(|mut c| {
        let mean = c.mean();
        let std = (c.variance() + EPS).sqrt();
        c.add_scalar_mut(-mean);
        c.unscale_mut(std);
        c.component_mul_assign(&norm.gamma);
        c += norm.beta;
    });
    x
}

fn silu<const DIM: usize, const SEQ: usize>(
    mut x: SMatrix<f32, DIM, SEQ>,
) -> SMatrix<f32, DIM, SEQ> {
    x.apply(|a| *a *= 1.0 / (1.0 + (-*a).exp()));
    x
}

#[derive(Debug)]
struct TimestepEmbedding<'a, const HALF: usize> {
    freqs: SVectorView<'a, f32, HALF>,
}

// t should be continuous (0, 1000)
fn timestep_embedding<const HALF: usize, const DIM: usize>(
    t: f32,
    embed: TimestepEmbedding<HALF>,
) -> SVector<f32, DIM> {
    debug_assert_eq!(DIM, 2*HALF);
    let scaled = embed.freqs.scale(t);
    let mut out = SVector::<f32, DIM>::zeros();
    out.rows_mut(0, HALF).copy_from(&scaled.map(f32::sin));
    out.rows_mut(HALF, HALF).copy_from(&scaled.map(f32::cos));
    out
}
