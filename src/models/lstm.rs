use std::iter::Map;

use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;
use primitiv::initializers as I;
use primitiv::functions as F;

/// Hand-written LSTM with input/forget/output gates and no peepholes.
/// Formulation:
///   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
///   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
///   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
///   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
///   c[t] = i * j + f * c[t-1]
///   h[t] = o * tanh(c[t])
pub struct LSTM {
    model: Model,
    pw: Parameter,
    pb: Parameter,
    w: Node,
    b: Node,
    h: Node,
    c: Node,
}

impl LSTM {
    pub fn new() -> Self {
        let mut m = LSTM {
            model: Model::new(),
            pw: Parameter::new(),
            pb: Parameter::new(),
            w: Node::new(),
            b: Node::new(),
            h: Node::new(),
            c: Node::new(),
        };
        m.model.add_parameter("w", &mut m.pw);
        m.model.add_parameter("b", &mut m.pb);
        m
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.pw.init_by_initializer(
            [4 * out_size, in_size + out_size],
            &I::Uniform::new(-0.1, 0.1),
        );
        self.pb.init_by_initializer(
            [4 * out_size],
            &I::Constant::new(1.0),
        );
    }

    /// Initializes internal values.
    pub fn restart(&mut self, init_c: Option<&Node>, init_h: Option<&Node>) {
        let out_size = self.pw.shape().at(0) / 4;
        self.w = F::parameter(&mut self.pw);
        self.b = F::parameter(&mut self.pb);
        self.c = init_c
            .and_then(|n| if n.valid() { Some(n.clone()) } else { None })
            .unwrap_or(F::zeros([out_size]));
        self.h = init_h
            .and_then(|n| if n.valid() { Some(n.clone()) } else { None })
            .unwrap_or(F::zeros([out_size]));
    }

    /// One step forwarding.
    pub fn forward<N: AsRef<Node>>(&mut self, x: N) -> &Node {
        let u = F::matmul(&self.w, F::concat(&vec![x.as_ref(), &self.h], 0)) + &self.b;
        let v = F::split(u, 0, 4);
        let i = F::sigmoid(&v[0]);
        let f = F::sigmoid(&v[1]);
        let o = F::sigmoid(&v[2]);
        let j = F::tanh(&v[3]);
        self.c = i * j + f * &self.c;
        self.h = o * F::tanh(&self.c);
        &self.h
    }

    pub fn get_c(&self) -> &Node {
        &self.c
    }

    pub fn get_h(&self) -> &Node {
        &self.h
    }
}

impl_model!(LSTM, model);

pub struct BiLSTM {
    model: Model,
    lstms: Vec<(LSTM, LSTM)>,
    dropout_rate: f32,
}

impl BiLSTM {
    pub fn new(n_layers: usize, dropout: f32) -> Self {
        assert!(n_layers > 0);
        let mut model = Model::new();
        let mut lstms = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let mut f_lstm = LSTM::new();
            let mut b_lstm = LSTM::new();
            model.add_submodel(&format!("f_lstm.{}", i), &mut f_lstm);
            model.add_submodel(&format!("b_lstm.{}", i), &mut b_lstm);
            lstms.push((f_lstm, b_lstm));
        }
        BiLSTM {
            model: model,
            lstms: lstms,
            dropout_rate: dropout,
        }
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        let mut iter = self.lstms.iter_mut();
        let &mut (ref mut first_f, ref mut first_b) = iter.next().unwrap();
        first_f.init(in_size, out_size);
        first_b.init(in_size, out_size);
        for &mut (ref mut lstm_f, ref mut lstm_b) in iter {
            lstm_f.init(in_size * 2, out_size);
            lstm_b.init(in_size * 2, out_size);
        }
    }

    /// Initializes internal values.
    pub fn restart(&mut self, init_states: Option<Vec<(Option<&Node>, Option<&Node>)>>) {
        let num_layers = self.lstms.len() * 2;
        let mut states = init_states.unwrap_or(vec![]);
        assert!(states.len() <= num_layers);
        for _ in 0..(num_layers - states.len()) {
            states.push((None, None));
        }
        for (i, (c, h)) in states.into_iter().enumerate() {
            let (ref mut lstm_f, ref mut lstm_b) = self.lstms[i / 2];
            if i % 2 == 0 {
                lstm_f.restart(c, h);
            } else {
                lstm_b.restart(c, h);
            }
        }
    }

    pub fn forward(&mut self, xs: &[Node], train: bool) -> Vec<Node> {
        let mut iter = self.lstms.iter_mut();
        let hs = {
            let &mut (ref mut lstm_f, ref mut lstm_b) = iter.next().unwrap();
            let xs_f = xs.iter();
            let xs_b = xs.iter().rev();
            let hs_f = xs_f.map(|x| lstm_f.forward(x).clone());
            let hs_b = xs_b.map(|x| lstm_b.forward(x).clone());
            hs_f.zip(hs_b.rev())
                .map(|(h_f, h_b)| F::concat(&[h_f, h_b], 0))
                .collect()
        };
        let mut xs_next: Vec<Node> = hs;
        let dropout_rate = self.dropout_rate;
        for &mut (ref mut lstm_f, ref mut lstm_b) in iter {
            let hs = {
                let xs_f = xs_next.iter();
                let xs_b = xs_next.iter().rev();
                let hs_f = xs_f.map(|x| {
                    lstm_f.forward(F::dropout(x, dropout_rate, train)).clone()
                });
                let hs_b = xs_b.map(|x| {
                    lstm_b.forward(F::dropout(x, dropout_rate, train)).clone()
                });
                hs_f.zip(hs_b.rev())
                    .map(|(h_f, h_b)| F::concat(&[h_f, h_b], 0))
                    .collect()
            };
            xs_next = hs;
        }
        xs_next
    }
}

impl_model!(BiLSTM, model);
