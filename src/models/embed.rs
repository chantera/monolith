use primitiv::Initializer;
use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;
use primitiv::initializers as I;
use primitiv::node_functions as F;

#[derive(Debug)]
pub struct Embed {
    model: Model,
    lookup: Parameter,
}

impl Embed {
    pub fn new() -> Self {
        let mut m = Embed {
            model: Model::new(),
            lookup: Parameter::new(),
        };
        m.model.add_parameter("lookup", &mut m.lookup);
        m
    }

    pub fn init(&mut self, vocab_size: usize, embed_size: u32) {
        self.init_by_initializer(vocab_size, embed_size, &I::Uniform::new(-0.1, 0.1));
    }

    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        vocab_size: usize,
        embed_size: u32,
        initializer: &I,
    ) {
        self.lookup.init_by_initializer(
            [embed_size, vocab_size as u32],
            initializer,
        );
    }

    pub fn init_by_values<Entries, Values>(&mut self, values: Entries)
    where
        Entries: AsRef<[Values]>,
        Values: AsRef<[f32]>,
    {
        assert!(values.as_ref().len() > 0);
        let vocab_size = values.as_ref().len();
        let embed_size = values.as_ref()[0].as_ref().len();
        let mut v = Vec::with_capacity(vocab_size * embed_size);
        values.as_ref().iter().for_each(|values| {
            v.extend(values.as_ref().iter().cloned())
        });
        self.lookup.init_by_values(
            [embed_size as u32, vocab_size as u32],
            &v,
        );
    }

    pub fn forward<Batch, IDs>(&mut self, xs: Batch) -> Vec<Node>
    where
        Batch: AsRef<[IDs]>,
        IDs: AsRef<[u32]>,
    {
        let lookup = F::parameter(&mut self.lookup);
        xs.as_ref()
            .iter()
            .map(|x| F::pick(&lookup, x.as_ref(), 1))
            .collect()
    }
}

impl_model!(Embed, model);
