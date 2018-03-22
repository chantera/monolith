use monolith::models::*;
use primitiv::node_functions as F;
use primitiv::Node;
use primitiv::Model;

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
        self.bilstm.reset(None);
        let hs = self.bilstm.forward(&xs, train);
        let ys = hs.into_iter().map(|h| self.mlp.forward(h, train)).collect();
        ys
    }

    pub fn loss<PosBatch, PosIDs>(&mut self, ys: &[Node], ts: PosBatch) -> Node
    where
        PosBatch: AsRef<[PosIDs]>,
        PosIDs: AsRef<[u32]>,
    {
        let losses: Vec<Node> = ts.as_ref()
            .iter()
            .zip(ys)
            .map(|(t, y)| F::softmax_cross_entropy_with_ids(y, t.as_ref(), 0))
            .collect();
        F::batch::mean(F::sum_nodes(&losses))
    }
}

impl_model!(Tagger, model);
