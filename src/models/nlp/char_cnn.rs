use std::f32::NEG_INFINITY;

use primitiv::functions as F;
use primitiv::Variable;

use models::{Conv2D, Embed, EmbedInitialize};

/// A Convolutional Neural Network that encodes character level information
///
/// See "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" (Ma and Hovy, 2016.)
/// http://www.aclweb.org/anthology/P16-1101
#[derive(Debug, Model, Serialize, Deserialize)]
pub struct CharCNN {
    #[primitiv(submodel)]
    embed: Embed,
    #[primitiv(submodel)]
    conv: Conv2D,
    pad_id: u32,
    pad_width: u32,
    dropout_rate: f32,
}

impl CharCNN {
    pub fn new(pad_id: u32, dropout_rate: f32) -> Self {
        CharCNN {
            embed: Embed::new(),
            conv: Conv2D::default(),
            pad_id,
            pad_width: 0,
            dropout_rate,
        }
    }

    pub fn init<I: EmbedInitialize>(&mut self, char_embed: I, out_size: u32, window_size: u32) {
        assert!(window_size % 2 == 1, "`window_size` must be odd value");
        self.embed.init_by(char_embed);
        let embed_size = self.embed.embed_size();
        self.pad_width = window_size / 2;
        let kernel = (embed_size, window_size);
        self.conv.padding = (0, self.pad_width);
        self.conv.stride = (embed_size, 1);
        self.conv.init(1, out_size, kernel);
    }

    pub fn forward<V: Variable, Seq, IDs>(&mut self, xs: &[Seq], train: bool) -> Vec<V>
    where
        Seq: AsRef<[IDs]>,
        IDs: AsRef<[u32]>,
    {
        self.forward_iter(xs.iter(), train).collect()
    }

    pub fn forward_iter<'a, V: Variable, It: 'a + Iterator<Item = Seq>, Seq, IDs>(
        &'a mut self,
        xs: It,
        train: bool,
    ) -> impl 'a + Iterator<Item = V>
    where
        Seq: AsRef<[IDs]>,
        IDs: AsRef<[u32]>,
    {
        xs.map(move |seq| self.forward_sequence(seq.as_ref(), train))
    }

    pub fn forward_sequence<V: Variable, IDs>(&mut self, xs: &[IDs], train: bool) -> V
    where
        IDs: AsRef<[u32]>,
    {
        let (ids, masks) = pad(xs, self.pad_width as usize, self.pad_id);
        let batch_size = xs.len() as u32;
        let out_len = ids[0].len() as u32;
        let xs: Vec<V> = self
            .embed
            .forward_iter(ids.into_iter())
            .map(|x: V| {
                F::concat(
                    F::batch::split(F::dropout(x, self.dropout_rate, train), out_len),
                    1,
                )
            })
            .collect();
        let hs = self.conv.forward(F::batch::concat(xs));
        let masks: Vec<f32> = masks.into_iter().flatten().collect();
        let masks: V = F::input(([1, out_len, 1], batch_size), &masks);
        let masks = F::broadcast(masks, 2, hs.shape().at(2));
        let ys = F::flatten(F::max(hs + masks, 1));
        ys
    }
}

#[inline]
fn pad<IDs: AsRef<[u32]>>(
    xs: &[IDs],
    pad_width: usize,
    pad_id: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let ids_with_len: Vec<(&[u32], usize)> = xs
        .as_ref()
        .iter()
        .map(|ids| {
            let ids = ids.as_ref();
            (ids, ids.len())
        })
        .collect();
    let max_len = ids_with_len.iter().max_by_key(|x| x.1).unwrap().1;
    let out_len = pad_width + max_len + pad_width;
    let mut masks: Vec<Vec<f32>> = Vec::with_capacity(out_len);
    let ids: Vec<Vec<u32>> = ids_with_len
        .into_iter()
        .map(|x| {
            let mut padded_ids = Vec::with_capacity(out_len);
            for _ in 0..pad_width {
                padded_ids.push(pad_id);
            }
            padded_ids.extend_from_slice(x.0);
            for _ in 0..(max_len - x.1 + pad_width) {
                padded_ids.push(pad_id);
            }
            let mut mask = vec![NEG_INFINITY; out_len];
            for i in pad_width..(pad_width + x.1) {
                mask[i] = 0.0;
            }
            masks.push(mask);
            padded_ids
        })
        .collect();
    (ids, masks)
}
