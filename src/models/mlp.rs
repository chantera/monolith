use primitiv::initializers as I;
use primitiv::node_functions as F;
use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;

#[derive(Debug, Serialize, Deserialize)]
struct LayerParameter {
    pw: Parameter,
    pb: Parameter,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MLP {
    layers: Vec<LayerParameter>,
    activation: Activate,
    dropout_rate: f32,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Activate {
    Sigmoid,
    Tanh,
    Relu,
    Softmax,
    Linear,
}

#[inline]
fn activate<N: AsRef<Node>>(activate: Activate, x: N) -> Node {
    let x = x.as_ref();
    match activate {
        Activate::Sigmoid => F::sigmoid(x),
        Activate::Tanh => F::tanh(x),
        Activate::Relu => F::relu(x),
        Activate::Softmax => F::softmax(x, 0),
        Activate::Linear => x.clone(),
    }
}

impl MLP {
    pub fn new(n_layers: usize, activation: Activate, dropout: f32) -> Self {
        if n_layers < 1 {
            panic!("number of layers must be greater than 0.");
        }
        let layers = (0..n_layers)
            .map(|_| LayerParameter {
                pw: Parameter::new(),
                pb: Parameter::new(),
            })
            .collect();
        MLP {
            layers: layers,
            activation: activation,
            dropout_rate: dropout,
        }
    }

    pub fn init(&mut self, units: &[u32], out_size: u32) {
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
            layer
                .pw
                .init_by_initializer([out, units[i]], &I::XavierUniform::new(1.0));
            layer.pb.init_by_initializer([out], &I::Constant::new(0.0));
        }
    }

    pub fn forward<N: AsRef<Node>>(&mut self, x: N, train: bool) -> Node {
        let num_layers = self.layers.len();
        let mut h = x.as_ref().clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let w = F::parameter(&mut layer.pw);
            let b = F::parameter(&mut layer.pb);
            if i < num_layers - 1 {
                h = F::dropout(
                    activate(self.activation, F::matmul(w, h) + b),
                    self.dropout_rate,
                    train,
                );
            } else {
                h = F::matmul(w, h) + b;
            }
        }
        h
    }

    pub fn forward_and_retrieve<N: AsRef<Node>>(
        &mut self,
        x: N,
        train: bool,
        hiddens: &mut Vec<Node>,
    ) -> Node {
        let num_layers = self.layers.len();
        let mut h = x.as_ref().clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let w = F::parameter(&mut layer.pw);
            let b = F::parameter(&mut layer.pb);
            if i < num_layers - 1 {
                h = F::dropout(
                    activate(self.activation, F::matmul(w, h) + b),
                    self.dropout_rate,
                    train,
                );
                hiddens.push(h.clone());
            } else {
                h = F::matmul(w, h) + b;
            }
        }
        h
    }
}

impl Model for MLP {
    fn register_parameters(&mut self) {
        let handle: *mut _ = self;
        unsafe {
            let model = &mut *handle;
            for (i, layer) in self.layers.iter_mut().enumerate() {
                model.add_parameter(&format!("{}.w", i), &mut layer.pw);
                model.add_parameter(&format!("{}.b", i), &mut layer.pb);
            }
        }
    }

    fn identifier(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        hasher.write(format!("{}-{:p}", "MLP", self).as_bytes());
        hasher.finish()
    }
}
