use std::f32::NEG_INFINITY;

use monolith::models::*;
use primitiv::node_functions as F;
use primitiv::Node;
use primitiv::Model;

const CHAR_EMBED_SIZE: u32 = 10;
const CHAR_PAD_ID: u32 = 1;
const CHAR_WINDOW_SIZE: u32 = 5;
const NUM_BILSTM_LAYERS: usize = 2;
const NUM_MLP_LAYERS: usize = 2;

#[derive(Debug)]
pub struct Tagger {
    model: Model,
    word_embed: Embed,
    char_cnn: Option<CharCNN>,
    bilstm: BiLSTM,
    mlp: MLP,
    dropout_rate: f32,
}

impl Tagger {
    pub fn new(use_char_cnn: bool, dropout: f32) -> Self {
        let mut m = Tagger {
            model: Model::new(),
            word_embed: Embed::new(),
            char_cnn: if use_char_cnn {
                Some(CharCNN::new(CHAR_PAD_ID, dropout))
            } else {
                None
            },
            bilstm: BiLSTM::new(NUM_BILSTM_LAYERS, dropout),
            mlp: MLP::new(NUM_MLP_LAYERS, Activate::Relu, dropout),
            dropout_rate: dropout,
        };
        m.model.add_submodel("word_embed", &mut m.word_embed);
        if let Some(ref mut char_cnn) = m.char_cnn {
            m.model.add_submodel("char_cnn", char_cnn);
        }
        m.model.add_submodel("bilstm", &mut m.bilstm);
        m.model.add_submodel("mlp", &mut m.mlp);
        m
    }

    pub fn init(
        &mut self,
        word_vocab_size: usize,
        word_embed_size: u32,
        char_vocab_size: usize,
        char_feature_size: u32,
        lstm_hidden_size: u32,
        mlp_unit: u32,
        out_size: usize,
    ) {
        self.word_embed.init(word_vocab_size, word_embed_size);
        self.word_embed.update_enabled = true;
        self.init_common(
            char_vocab_size,
            char_feature_size,
            lstm_hidden_size,
            mlp_unit,
            out_size,
        );
    }

    pub fn init_by_values<Entries, Values>(
        &mut self,
        word_embed: Entries,
        char_vocab_size: usize,
        char_feature_size: u32,
        lstm_hidden_size: u32,
        mlp_unit: u32,
        out_size: usize,
    ) where
        Entries: AsRef<[Values]>,
        Values: AsRef<[f32]>,
    {
        self.word_embed.init_by_values(word_embed);
        self.word_embed.update_enabled = false;
        self.init_common(
            char_vocab_size,
            char_feature_size,
            lstm_hidden_size,
            mlp_unit,
            out_size,
        );
    }

    fn init_common(
        &mut self,
        char_vocab_size: usize,
        char_feature_size: u32,
        lstm_hidden_size: u32,
        mlp_unit: u32,
        out_size: usize,
    ) {
        let mut bilstm_in_size = self.word_embed.embed_size();
        if let Some(ref mut char_cnn) = self.char_cnn {
            char_cnn.init(
                char_vocab_size,
                CHAR_EMBED_SIZE,
                char_feature_size,
                CHAR_WINDOW_SIZE,
            );
            bilstm_in_size += char_feature_size;
        }
        self.bilstm.init(bilstm_in_size, lstm_hidden_size);
        self.mlp.init(
            &[lstm_hidden_size * 2, mlp_unit],
            out_size as u32,
        );
    }

    pub fn forward<WordBatch, CharsBatch, Chars, WordIDs, CharIDs>(
        &mut self,
        words: WordBatch,
        chars: CharsBatch,
        train: bool,
    ) -> Vec<Node>
    where
        WordBatch: AsRef<[WordIDs]>,
        CharsBatch: AsRef<[Chars]>,
        Chars: AsRef<[CharIDs]>,
        WordIDs: AsRef<[u32]>,
        CharIDs: AsRef<[u32]>,
    {
        let dropout_rate = self.dropout_rate;
        let xs_word = self.word_embed.forward(words);
        let xs = if let Some(ref mut char_cnn) = self.char_cnn {
            let xs_char = char_cnn.forward(chars, train);
            xs_word
                .into_iter()
                .zip(xs_char.into_iter())
                .map(|(x_w, x_c)| {
                    F::concat(
                        [
                            F::dropout(x_w, dropout_rate, train),
                            F::dropout(x_c, dropout_rate, train),
                        ],
                        0,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            xs_word
                .into_iter()
                .map(|x_w| F::dropout(x_w, dropout_rate, train))
                .collect::<Vec<_>>()
        };
        let hs = self.bilstm.forward(&xs, None, train);
        let ys = hs.into_iter().map(|h| self.mlp.forward(h, train)).collect();
        ys
    }

    pub fn loss<PosBatch, PosIDs>(&mut self, ys: &[Node], ts: PosBatch) -> Node
    where
        PosBatch: AsRef<[PosIDs]>,
        PosIDs: AsRef<[u32]>,
    {
        let batch_size = ys[0].shape().batch();
        let losses: Vec<Node> = ts.as_ref()
            .iter()
            .zip(ys)
            .map(|(t, y)| {
                F::batch::sum(F::softmax_cross_entropy_with_ids(y, t.as_ref(), 0))
            })
            .collect();
        F::sum_nodes(&losses) / batch_size
    }

    pub fn accuracy<PosBatch, PosIDs>(&mut self, ys: &[Node], ts: PosBatch) -> (u32, u32)
    where
        PosBatch: AsRef<[PosIDs]>,
        PosIDs: AsRef<[u32]>,
    {
        let mut correct = 0;
        let mut total = 0;
        for (y_batch, t_batch) in ys.iter().zip(ts.as_ref()) {
            for (y, t) in y_batch.argmax(0).iter().zip(t_batch.as_ref()) {
                total += 1;
                if y == t {
                    correct += 1;
                }
            }
        }
        (correct, total)
    }
}

impl_model!(Tagger, model);

#[derive(Debug)]
pub struct TaggerBuilder<'a> {
    dropout_rate: f32,
    word_vocab_size: usize,
    word_embed_size: u32,
    word_embed: Option<&'a Vec<Vec<f32>>>,
    use_char_cnn: bool,
    char_vocab_size: usize,
    char_feature_size: u32,
    lstm_hidden_size: u32,
    mlp_unit: u32,
    out_size: Option<usize>,
}

impl<'a> TaggerBuilder<'a> {
    pub fn new() -> Self {
        TaggerBuilder {
            dropout_rate: 0.5,
            word_vocab_size: 60000,
            word_embed_size: 100,
            word_embed: None,
            use_char_cnn: false,
            char_vocab_size: 128,
            char_feature_size: 50,
            lstm_hidden_size: 200,
            mlp_unit: 200,
            out_size: None,
        }
    }

    pub fn build(self) -> Tagger {
        if self.out_size.is_none() {
            panic!("out_size must be set before builder.build() is called.");
        }
        let mut tagger = Tagger::new(self.use_char_cnn, self.dropout_rate);
        match self.word_embed {
            Some(values) => {
                tagger.init_by_values(
                    values,
                    self.char_vocab_size,
                    self.char_feature_size,
                    self.lstm_hidden_size,
                    self.mlp_unit,
                    self.out_size.unwrap(),
                );
            }
            None => {
                tagger.init(
                    self.word_vocab_size,
                    self.word_embed_size,
                    self.char_vocab_size,
                    self.char_feature_size,
                    self.lstm_hidden_size,
                    self.mlp_unit,
                    self.out_size.unwrap(),
                );
            }
        }
        tagger
    }

    pub fn word(mut self, vocab_size: usize, embed_size: u32) -> Self {
        self.word_vocab_size = vocab_size;
        self.word_embed_size = embed_size;
        self
    }

    pub fn word_embed(mut self, values: &'a Vec<Vec<f32>>) -> Self {
        self.word_embed = Some(values);
        self
    }

    pub fn char(mut self, vocab_size: usize, feature_size: u32) -> Self {
        self.char_vocab_size = vocab_size;
        self.char_feature_size = feature_size;
        self.use_char_cnn = true;
        self
    }

    pub fn dropout(mut self, p: f32) -> Self {
        self.dropout_rate = p;
        self
    }

    pub fn lstm(mut self, hidden_size: u32) -> Self {
        self.lstm_hidden_size = hidden_size;
        self
    }

    pub fn mlp(mut self, unit: u32) -> Self {
        self.mlp_unit = unit;
        self
    }

    pub fn out(mut self, size: usize) -> Self {
        self.out_size = Some(size);
        self
    }
}

#[derive(Debug)]
pub struct CharCNN {
    model: Model,
    embed: Embed,
    conv: Conv2D,
    pad_id: u32,
    pad_width: u32,
    dropout_rate: f32,
}

impl CharCNN {
    pub fn new(pad_id: u32, dropout: f32) -> Self {
        let mut m = CharCNN {
            model: Model::new(),
            embed: Embed::new(),
            conv: Conv2D::default(),
            pad_id: pad_id,
            pad_width: 0,
            dropout_rate: dropout,
        };
        m.model.add_submodel("embed", &mut m.embed);
        m.model.add_submodel("conv", &mut m.conv);
        m
    }

    pub fn init(&mut self, vocab_size: usize, embed_size: u32, out_size: u32, window_size: u32) {
        assert!(window_size % 2 == 1, "`window_size` must be odd value");
        self.embed.init(vocab_size, embed_size);
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
        xs = xs.into_iter()
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

impl_model!(CharCNN, model);

#[inline]
fn pad<Sequence, IDs>(xs: Sequence, pad_width: usize, pad_id: u32) -> (Vec<Vec<u32>>, Vec<f32>)
where
    Sequence: AsRef<[IDs]>,
    IDs: AsRef<[u32]>,
{
    let ids_with_len: Vec<(&[u32], usize)> = xs.as_ref()
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
