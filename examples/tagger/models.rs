use monolith::models::*;
use primitiv::functions as F;
use primitiv::{Node, Variable};

const CHAR_EMBED_SIZE: u32 = 10;
const CHAR_PAD_ID: u32 = 1;
const CHAR_WINDOW_SIZE: u32 = 5;
const NUM_BILSTM_LAYERS: usize = 2;
const NUM_MLP_LAYERS: usize = 2;

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct Tagger<V: Variable> {
    #[primitiv(submodel)]
    word_embed: Embed,
    #[primitiv(submodel)]
    char_cnn: Option<nlp::CharCNN>,
    #[primitiv(submodel)]
    bilstm: BiLSTM<V>,
    #[primitiv(submodel)]
    mlp: MLP,
    dropout_rate: f32,
}

impl<V: Variable> Tagger<V> {
    pub fn new(use_char_cnn: bool, dropout_rate: f32) -> Self {
        Tagger {
            word_embed: Embed::new(),
            char_cnn: if use_char_cnn {
                Some(nlp::CharCNN::new(CHAR_PAD_ID, dropout_rate))
            } else {
                None
            },
            bilstm: BiLSTM::new(NUM_BILSTM_LAYERS, dropout_rate),
            mlp: MLP::new(NUM_MLP_LAYERS, Activation::Relu, dropout_rate),
            dropout_rate,
        }
    }

    pub fn init<I: EmbedInitialize>(
        &mut self,
        word_embed: I,
        char_vocab_size: u32,
        char_feature_size: u32,
        lstm_hidden_size: u32,
        mlp_unit: u32,
        out_size: u32,
    ) {
        self.word_embed.init_by(word_embed);
        let mut bilstm_in_size = self.word_embed.embed_size();
        if let Some(ref mut char_cnn) = self.char_cnn {
            char_cnn.init(
                (char_vocab_size, CHAR_EMBED_SIZE),
                char_feature_size,
                CHAR_WINDOW_SIZE,
            );
            bilstm_in_size += char_feature_size;
        }
        self.bilstm.init(bilstm_in_size, lstm_hidden_size);
        self.mlp
            .init(&[lstm_hidden_size * 2, mlp_unit], out_size as u32);
    }

    pub fn forward<WordIDs, Chars, CharIDs>(
        &mut self,
        words: &[WordIDs],
        chars: &[Chars],
        train: bool,
    ) -> Vec<V>
    where
        WordIDs: AsRef<[u32]>,
        Chars: AsRef<[CharIDs]>,
        CharIDs: AsRef<[u32]>,
    {
        let dropout_rate = self.dropout_rate;
        let xs_word = self.word_embed.forward_iter(words.iter());
        let xs: Vec<V> = if let Some(ref mut char_cnn) = self.char_cnn {
            xs_word
                .zip(char_cnn.forward_iter(chars.iter(), train))
                .map(|(x_w, x_c): (V, V)| {
                    F::concat(
                        [
                            F::dropout(x_w, dropout_rate, train),
                            F::dropout(x_c, dropout_rate, train),
                        ],
                        0,
                    )
                })
                .collect()
        } else {
            xs_word
                .map(|x_w| F::dropout(x_w, dropout_rate, train))
                .collect()
        };
        let hs = self.bilstm.forward(xs, None, train);
        let ys = hs.into_iter().map(|h| self.mlp.forward(h, train)).collect();
        ys
    }

    pub fn loss<IDs: AsRef<[u32]>>(&mut self, ys: &[V], ts: &[IDs]) -> V {
        let batch_size = ys[0].shape().batch();
        let losses: Vec<V> = ts
            .iter()
            .zip(ys)
            .map(|(t, y)| F::batch::sum(F::softmax_cross_entropy_with_ids(y, t.as_ref(), 0)))
            .collect();
        F::sum_vars(&losses) / batch_size
    }

    pub fn accuracy<IDs: AsRef<[u32]>>(&mut self, ys: &[V], ts: &[IDs]) -> (u32, u32) {
        let mut correct = 0;
        let mut total = 0;
        for (y_batch, t_batch) in ys.iter().zip(ts) {
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

#[derive(Debug)]
pub struct TaggerBuilder<'a> {
    dropout_rate: f32,
    word_vocab_size: u32,
    word_embed_size: u32,
    word_embed: Option<&'a Vec<Vec<f32>>>,
    use_char_cnn: bool,
    char_vocab_size: u32,
    char_feature_size: u32,
    lstm_hidden_size: u32,
    mlp_unit: u32,
    out_size: Option<u32>,
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

    pub fn build(self) -> Tagger<Node> {
        if self.out_size.is_none() {
            panic!("out_size must be set before builder.build() is called.");
        }
        let mut tagger = Tagger::new(self.use_char_cnn, self.dropout_rate);
        match self.word_embed {
            Some(values) => {
                tagger.init(
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
                    (self.word_vocab_size, self.word_embed_size),
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

    pub fn word(mut self, vocab_size: u32, embed_size: u32) -> Self {
        self.word_vocab_size = vocab_size;
        self.word_embed_size = embed_size;
        self
    }

    pub fn word_embed(mut self, values: &'a Vec<Vec<f32>>) -> Self {
        self.word_embed = Some(values);
        self
    }

    pub fn char(mut self, vocab_size: u32, feature_size: u32) -> Self {
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

    pub fn out(mut self, size: u32) -> Self {
        self.out_size = Some(size);
        self
    }
}
