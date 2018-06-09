use primitiv::functions as F;
use primitiv::Initializer;
use primitiv::Variable;

use models::Linear;

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct MLP {
    #[primitiv(submodel)]
    layers: Vec<Linear>,
    activation: Activation,
    dropout_rate: f32,
}

impl MLP {
    pub fn new(n_layers: usize, activation: Activation, dropout_rate: f32) -> Self {
        if n_layers < 1 {
            panic!("number of layers must be greater than 0.");
        }
        MLP {
            layers: (0..n_layers).map(|_| Linear::new(true)).collect(),
            activation,
            dropout_rate,
        }
    }

    pub fn init(&mut self, units: &[u32], out_size: u32) {
        self.init_by_initializer(units, out_size, &Self::default_initializer())
    }

    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        units: &[u32],
        out_size: u32,
        initializer: &I,
    ) {
        let num_layers = self.layers.len();
        assert_eq!(
            num_layers,
            units.len(),
            "the number of unit values must be equal to that of the layers"
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let out = if i < num_layers - 1 {
                units[i + 1]
            } else {
                out_size
            };
            layer.init_by_initializer(units[i], out, initializer);
        }
    }

    pub fn forward<V: Variable, X: AsRef<V>>(&mut self, x: X, train: bool) -> V {
        let num_layers = self.layers.len();
        let mut h = x.as_ref().clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < num_layers - 1 {
                h = F::dropout(
                    self.activation.forward(layer.forward(h)),
                    self.dropout_rate,
                    train,
                );
            } else {
                h = layer.forward(h)
            }
        }
        h
    }

    pub fn default_initializer() -> impl Initializer {
        Linear::default_initializer()
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Tanh,
    Relu,
    Softmax,
    Linear,
}

impl Activation {
    fn forward<V: Variable, X: AsRef<V>>(&self, x: X) -> V {
        let x = x.as_ref();
        match *self {
            Activation::Sigmoid => F::sigmoid(x),
            Activation::Tanh => F::tanh(x),
            Activation::Relu => F::relu(x),
            Activation::Softmax => F::softmax(x, 0),
            Activation::Linear => x.clone(),
        }
    }
}
