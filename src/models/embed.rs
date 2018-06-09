use primitiv::functions as F;
use primitiv::initializers::Uniform;
use primitiv::Initializer;
use primitiv::Parameter;
use primitiv::Variable;

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct Embed {
    lookup: Parameter,
    update_enabled: bool,
}

impl Embed {
    pub fn new() -> Self {
        Embed {
            lookup: Parameter::new(),
            update_enabled: true,
        }
    }

    pub fn init(&mut self, vocab_size: u32, embed_size: u32) {
        self.init_by_initializer(vocab_size, embed_size, &Self::default_initializer());
    }

    pub fn init_by<I: EmbedInitialize>(&mut self, initializer: I) {
        initializer.initialize(self);
    }

    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        vocab_size: u32,
        embed_size: u32,
        initializer: &I,
    ) {
        self.lookup
            .init_by_initializer([embed_size, vocab_size], initializer);
    }

    pub fn init_by_values<E: AsRef<[f32]>>(&mut self, values: &[E]) {
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

    pub fn forward<V: Variable, IDs: AsRef<[u32]>>(&mut self, xs: &[IDs]) -> Vec<V> {
        self.forward_iter(xs.iter()).collect()
    }

    pub fn forward_iter<V: Variable, It: Iterator<Item = IDs>, IDs: AsRef<[u32]>>(
        &mut self,
        xs: It,
    ) -> impl Iterator<Item = V> {
        let mut lookup: V = F::parameter(&mut self.lookup);
        if !self.update_enabled {
            lookup = F::stop_gradient(lookup);
        }
        xs.map(move |x| F::pick(&lookup, x.as_ref(), 1))
    }

    pub fn embed_size(&self) -> u32 {
        self.lookup.shape().at(0)
    }

    pub fn vocab_size(&self) -> u32 {
        self.lookup.shape().at(1)
    }

    pub fn is_enabled_update(&self) -> bool {
        self.update_enabled
    }

    pub fn enable_update(&mut self) {
        self.update_enabled = true;
    }

    pub fn disable_update(&mut self) {
        self.update_enabled = false;
    }

    pub fn default_initializer() -> impl Initializer {
        Uniform::new(-0.1, 0.1)
    }
}

pub trait EmbedInitialize {
    fn initialize(&self, embed: &mut Embed);
}

impl EmbedInitialize for (u32, u32) {
    fn initialize(&self, embed: &mut Embed) {
        embed.init(self.0, self.1);
        embed.enable_update()
    }
}

impl<I: Initializer> EmbedInitialize for (u32, u32, I) {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_initializer(self.0, self.1, &self.2);
        embed.enable_update()
    }
}

impl<V: AsRef<[f32]>> EmbedInitialize for Vec<V> {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_values(self);
        embed.disable_update()
    }
}

impl<'a, V: AsRef<[f32]>> EmbedInitialize for &'a Vec<V> {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_values(self);
        embed.disable_update()
    }
}

impl<'a, V: AsRef<[f32]>> EmbedInitialize for &'a [V] {
    fn initialize(&self, embed: &mut Embed) {
        embed.init_by_values(self);
        embed.disable_update()
    }
}
