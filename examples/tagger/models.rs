use monolith::models::*;
use primitiv::node_functions as F;
use primitiv::Node;
use primitiv::Model;

pub struct Tagger {
    model: Model,
    word_embed: Embed<Node>,
    char_embed: Embed<Node>,
    bilstm: BiLSTM,
    // mlp: MLP,
    dropout_rate: f32,
}

impl Tagger {
    pub fn new(dropout: f32) -> Self {
        let mut m = Tagger {
            model: Model::new(),
            word_embed: Embed::new(),
            char_embed: Embed::new(),
            bilstm: BiLSTM::new(2, dropout),
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
    ) {
        self.word_embed.init(word_vocab_size, word_embed_size);
        self.char_embed.init(char_vocab_size, char_embed_size);
        self.bilstm.init(word_embed_size, lstm_hidden_size);
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
        let mut xs = self.word_embed.forward(words);
        if train {
            xs = xs.into_iter()
                .map(|x| F::dropout(x, self.dropout_rate, true))
                .collect::<Vec<_>>();
        }
        self.bilstm.reset(None);
        let hs = self.bilstm.forward(&xs, train);
        hs
    }
}

impl_model!(Tagger, model);
