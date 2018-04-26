use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;
use primitiv::initializers as I;
use primitiv::node_functions as F;

#[derive(Debug)]
pub struct MLP {
    model: Model,
    layers: Vec<(Parameter, Parameter)>,
    activation: Activate,
    dropout_rate: f32,
}

#[derive(Copy, Clone, Debug)]
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
        let mut model = Model::new();
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let mut pw = Parameter::new();
            let mut pb = Parameter::new();
            model.add_parameter(&format!("{}.pw", i), &mut pw);
            model.add_parameter(&format!("{}.pb", i), &mut pb);
            layers.push((pw, pb));
        }
        MLP {
            model: model,
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
            layer.0.init_by_initializer(
                [out, units[i]],
                &I::XavierUniform::new(1.0),
            );
            layer.1.init_by_initializer([out], &I::Constant::new(0.0));
        }
    }

    pub fn forward<N: AsRef<Node>>(&mut self, x: N, train: bool) -> Node {
        let num_layers = self.layers.len();
        let mut h = x.as_ref().clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let w = F::parameter(&mut layer.0);
            let b = F::parameter(&mut layer.1);
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
}

impl_model!(MLP, model);
