use std::f32::NEG_INFINITY;

use primitiv::node_functions as F;
use primitiv::Node;

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
    pub fn new(pad_id: u32, dropout: f32) -> Self {
        CharCNN {
            embed: Embed::new(),
            conv: Conv2D::default(),
            pad_id: pad_id,
            pad_width: 0,
            dropout_rate: dropout,
        }
    }

    pub fn init<I: EmbedInitialize>(&mut self, char_embed: I, out_size: u32, window_size: u32) {
        assert!(window_size % 2 == 1, "`window_size` must be odd value");
        self.embed.init_from(char_embed);
        let embed_size = self.embed.embed_size();
        self.pad_width = window_size / 2;
        let kernel = (embed_size, window_size);
        self.conv.padding = (0, self.pad_width);
        self.conv.stride = (embed_size, 1);
        self.conv.init(1, out_size, kernel);
    }

    pub fn forward<Batch, Sequence, IDs>(&mut self, xs: Batch, train: bool) -> Vec<Node>
    where
        Batch: AsRef<[Sequence]>,
        Sequence: AsRef<[IDs]>,
        IDs: AsRef<[u32]>,
    {
        xs.as_ref()
            .iter()
            .map(|seq| self.forward_sequence(seq, train))
            .collect()
    }

    pub fn forward_sequence<Sequence, IDs>(&mut self, xs: Sequence, train: bool) -> Node
    where
        Sequence: AsRef<[IDs]>,
        IDs: AsRef<[u32]>,
    {
        let (ids, mask) = pad(xs, self.pad_width as usize, self.pad_id);
        let out_len = mask.len();
        let mut xs = self.embed.forward(ids);
        xs = xs
            .into_iter()
            .map(|x| {
                F::concat(
                    F::batch::split(F::dropout(x, self.dropout_rate, train), out_len as u32),
                    1,
                )
            })
            .collect::<Vec<_>>();
        let xs = F::batch::concat(xs);
        let hs = self.conv.forward(xs);
        let mask = F::broadcast(F::input([1, out_len as u32, 1], &mask), 2, hs.shape().at(2));
        let ys = F::flatten(F::max(hs + mask, 1));
        ys
    }
}

#[inline]
fn pad<IDs: AsRef<[u32]>>(
    xs: impl AsRef<[IDs]>,
    pad_width: usize,
    pad_id: u32,
) -> (Vec<Vec<u32>>, Vec<f32>) {
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
            padded_ids
        })
        .collect();

    let mut mask = Vec::with_capacity(out_len);
    for _ in 0..pad_width {
        mask.push(NEG_INFINITY);
    }
    for _ in 0..(max_len) {
        mask.push(0.0);
    }
    for _ in 0..pad_width {
        mask.push(NEG_INFINITY);
    }

    (ids, mask)
}
