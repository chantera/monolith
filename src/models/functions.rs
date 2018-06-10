use primitiv::functions as F;
use primitiv::Variable;

pub fn pad_sequence<V: Variable>(xs: Vec<V>, pad_value: f32) -> (V, Vec<usize>) {
    let shape = xs[0].shape();
    let batch_size = shape.batch();
    let mut lengths = Vec::with_capacity(xs.len());
    let xs_padded: Vec<V> = xs
        .into_iter()
        .map(|x| {
            let n = x.shape().batch();
            lengths.push(n as usize);
            if n == batch_size {
                x
            } else {
                let pad: V = F::constant(shape.resize_batch(batch_size - n), pad_value);
                F::batch::concat([x, pad])
            }
        })
        .collect();
    (F::concat(xs_padded, 1), lengths)
}

pub fn transpose_sequence<V: Variable>(xs: Vec<V>) -> Vec<V> {
    let batch_size = xs[0].shape().batch() as usize;
    let mut lengths = vec![xs.len(); batch_size];
    let mut end = batch_size;
    let xs: Vec<V> = xs
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            let mut s = x.shape();
            let len = s.batch() as usize;
            if len < end {
                for l in lengths[len..end].iter_mut() {
                    *l = i;
                }
                end = len;
            }
            let diff = batch_size - len;
            if diff > 0 {
                s.update_batch(diff as u32);
                F::batch::concat(&[x, F::zeros(s)])
            } else {
                x
            }
        })
        .collect();
    let xs_transposed = F::batch::split(F::transpose(F::concat(&xs, 1)), batch_size as u32);
    xs_transposed
        .into_iter()
        .zip(lengths.into_iter())
        .map(|(x, len)| F::slice(x, 0, 0, len as u32))
        .collect()
}
