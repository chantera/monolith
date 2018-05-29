use primitiv::initializers::Uniform;
use primitiv::node_functions as F;
use primitiv::Initializer;
use primitiv::Node;
use primitiv::Parameter;

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct Embed {
    lookup: Parameter,
    pub update_enabled: bool,
}

impl Embed {
    pub fn new() -> Self {
        Embed {
            lookup: Parameter::new(),
            update_enabled: true,
        }
    }

    pub fn init(&mut self, vocab_size: usize, embed_size: u32) {
        self.init_by_initializer(vocab_size, embed_size, &Uniform::new(-0.1, 0.1));
    }

    pub fn init_from(&mut self, initializer: impl EmbedInitialize) {
        initializer.initialize(self);
    }

    pub fn init_by_initializer(
        &mut self,
        vocab_size: usize,
        embed_size: u32,
        initializer: &impl Initializer,
    ) {
        self.lookup
            .init_by_initializer([embed_size, vocab_size as u32], initializer);
    }

    pub fn init_by_values<V: AsRef<[f32]>>(&mut self, values: impl AsRef<[V]>) {
        debug_assert!(values.as_ref().len() > 0);
        let vocab_size = values.as_ref().len();
        let embed_size = values.as_ref()[0].as_ref().len();
        let mut v = Vec::with_capacity(vocab_size * embed_size);
        values
            .as_ref()
            .iter()
            .for_each(|values| v.extend(values.as_ref().iter().cloned()));
        self.lookup
            .init_by_values([embed_size as u32, vocab_size as u32], &v);
    }

    pub fn forward<IDs: AsRef<[u32]>>(&mut self, xs: impl AsRef<[IDs]>) -> Vec<Node> {
        self.forward_iter(xs.as_ref().iter()).collect()
    }

    pub fn forward_iter<IDs: AsRef<[u32]>>(
        &mut self,
        xs: impl Iterator<Item = IDs>,
    ) -> impl Iterator<Item = Node> {
        let mut lookup = F::parameter(&mut self.lookup);
        if !self.update_enabled {
            lookup = F::stop_gradient(lookup);
        }
        xs.map(move |x| F::pick(&lookup, x.as_ref(), 1))
    }

    pub fn initialized(&self) -> bool {
        self.lookup.valid()
    }

    pub fn embed_size(&self) -> u32 {
        self.lookup.shape().at(0)
    }

    pub fn vocab_size(&self) -> usize {
        self.lookup.shape().at(1) as usize
    }
}

pub trait EmbedInitialize {
    fn initialize(&self, embed: &mut Embed);
}

impl EmbedInitialize for (usize, u32) {
    fn initialize(&self, embed: &mut Embed) {
        embed.init(self.0, self.1);
        embed.update_enabled = true;
    }
}

impl<I: Initializer> EmbedInitialize for (usize, u32, I) {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_initializer(self.0, self.1, &self.2);
        embed.update_enabled = true;
    }
}

impl<V: AsRef<[f32]>> EmbedInitialize for Vec<V> {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_values(self);
        embed.update_enabled = false;
    }
}

impl<'a, V: AsRef<[f32]>> EmbedInitialize for &'a Vec<V> {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_values(self);
        embed.update_enabled = false;
    }
}

impl<'a, V: AsRef<[f32]>> EmbedInitialize for &'a [V] {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_values(self);
        embed.update_enabled = false;
    }
}
