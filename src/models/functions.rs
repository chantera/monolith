use primitiv::functions as F;
use primitiv::Variable;

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
