use monolith::models::*;
use primitiv::node_functions as F;
use primitiv::Node;
use primitiv::Model;

#[derive(Debug)]
pub struct Tagger {
    model: Model,
    word_embed: Embed,
    char_embed: Embed,
    bilstm: BiLSTM,
    mlp: MLP,
    dropout_rate: f32,
}

impl Tagger {
    pub fn new(dropout: f32) -> Self {
        let mut m = Tagger {
            model: Model::new(),
            word_embed: Embed::new(),
            char_embed: Embed::new(),
            bilstm: BiLSTM::new(2, dropout),
            mlp: MLP::new(2, Activate::Relu, dropout),
            dropout_rate: dropout,
        };
        m.model.add_submodel("word_embed", &mut m.word_embed);
        m.model.add_submodel("char_embed", &mut m.char_embed);
        m.model.add_submodel("bilstm", &mut m.bilstm);
        m.model.add_submodel("mlp", &mut m.mlp);
        m
    }

    pub fn init(
        &mut self,
        word_vocab_size: usize,
        word_embed_size: u32,
        char_vocab_size: usize,
        char_embed_size: u32,
        lstm_hidden_size: u32,
        mlp_unit: u32,
        out_size: usize,
    ) {
        self.word_embed.init(word_vocab_size, word_embed_size);
        self.char_embed.init(char_vocab_size, char_embed_size);
        self.bilstm.init(word_embed_size, lstm_hidden_size);
        self.mlp.init(
            &[lstm_hidden_size * 2, mlp_unit],
            out_size as u32,
        );
    }

    pub fn forward<WordBatch, CharsBatch, Chars, WordIDs, CharIDs>(
        &mut self,
        words: WordBatch,
        _chars: CharsBatch,
        train: bool,
    ) -> Vec<Node>
    where
        WordBatch: AsRef<[WordIDs]>,
        CharsBatch: AsRef<[Chars]>,
        Chars: AsRef<[CharIDs]>,
        WordIDs: AsRef<[u32]>,
        CharIDs: AsRef<[u32]>,
    {
        let mut xs = self.word_embed.forward(words);
        if train {
            xs = xs.into_iter()
                .map(|x| F::dropout(x, self.dropout_rate, true))
                .collect::<Vec<_>>();
        }
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
pub struct TaggerBuilder {
    dropout_rate: f32,
    word_vocab_size: usize,
    word_embed_size: u32,
    char_vocab_size: usize,
    char_embed_size: u32,
    lstm_hidden_size: u32,
    mlp_unit: u32,
    out_size: Option<usize>,
}

impl TaggerBuilder {
    pub fn new() -> Self {
        TaggerBuilder {
            dropout_rate: 0.5,
            word_vocab_size: 60000,
            word_embed_size: 100,
            char_vocab_size: 128,
            char_embed_size: 32,
            lstm_hidden_size: 200,
            mlp_unit: 200,
            out_size: None,
        }
    }

    pub fn build(self) -> Tagger {
        if self.out_size.is_none() {
            panic!("out_size must be set before builder.build() is called.");
        }
        let mut tagger = Tagger::new(self.dropout_rate);
        tagger.init(
            self.word_vocab_size,
            self.word_embed_size,
            self.char_vocab_size,
            self.char_embed_size,
            self.lstm_hidden_size,
            self.mlp_unit,
            self.out_size.unwrap(),
        );
        tagger
    }

    pub fn word(mut self, vocab_size: usize, embed_size: u32) -> Self {
        self.word_vocab_size = vocab_size;
        self.word_embed_size = embed_size;
        self
    }

    pub fn char(mut self, vocab_size: usize, embed_size: u32) -> Self {
        self.char_vocab_size = vocab_size;
        self.char_embed_size = embed_size;
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
