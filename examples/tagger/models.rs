use monolith::models::*;
use primitiv::Node;
use primitiv::Model;

pub struct Tagger {
    model: Model,
    word_embed: Embed,
    char_embed: Embed,
    // bilstm: BiLSTM,
    // mlp: MLP,
}

impl Tagger {
    pub fn new() -> Self {
        let mut m = Tagger {
            model: Model::new(),
            // dropout_rate: DROPOUT_RATE,
            word_embed: Embed::new(),
            char_embed: Embed::new(),
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
    ) {
        self.word_embed.init(word_vocab_size, word_embed_size);
        self.char_embed.init(char_vocab_size, char_embed_size);
    }

    pub fn forward<WordBatch, CharsBatch, Chars, WordIDs, CharIDs>(
        &mut self,
        words: WordBatch,
        chars: CharsBatch,
    ) -> Vec<Node>
    where
        WordBatch: AsRef<[WordIDs]>,
        CharsBatch: AsRef<[Chars]>,
        Chars: AsRef<[CharIDs]>,
        WordIDs: AsRef<[u32]>,
        CharIDs: AsRef<[u32]>,
    {
        let xs = self.word_embed.forward(words);
        xs
    }
}

impl_model!(Tagger, model);
