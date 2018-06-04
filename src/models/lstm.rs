use primitiv::initializers as I;
use primitiv::node_functions as F;
use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;
use primitiv::Shape;

/// Hand-written LSTM with input/forget/output gates and no peepholes.
/// Formulation:
///   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
///   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
///   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
///   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
///   c[t] = i * j + f * c[t-1]
///   h[t] = o * tanh(c[t])
#[derive(Debug, Model, Serialize, Deserialize)]
pub struct LSTM {
    pw: Parameter,
    pb: Parameter,
    w: Node,
    b: Node,
    h: Node,
    c: Node,
}

impl LSTM {
    pub fn new() -> Self {
        LSTM {
            pw: Parameter::new(),
            pb: Parameter::new(),
            w: Node::new(),
            b: Node::new(),
            h: Node::new(),
            c: Node::new(),
        }
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.pw.init_by_initializer(
            [4 * out_size, in_size + out_size],
            &I::Uniform::new(-0.1, 0.1),
        );
        let mut bias_values = vec![0.0; 4 * out_size as usize];
        for i in (out_size as usize)..(2 * out_size as usize) {
            bias_values[i] = 1.0;
        }
        self.pb.init_by_values([4 * out_size], &bias_values);
    }

    /// Initializes internal values.
    pub fn reset(&mut self, init_c: Option<Node>, init_h: Option<Node>) {
        let out_size = self.pw.shape().at(0) / 4;
        self.w = F::parameter(&mut self.pw);
        self.b = F::parameter(&mut self.pb);
        self.c = init_c
            .and_then(|n| if n.valid() { Some(n) } else { None })
            .unwrap_or(F::zeros([out_size]));
        self.h = init_h
            .and_then(|n| if n.valid() { Some(n) } else { None })
            .unwrap_or(F::zeros([out_size]));
    }

    /// One step forwarding.
    pub fn forward<N: AsRef<Node>>(&mut self, x: N) -> Node {
        let x = x.as_ref();
        let (lstm_in, h_rest): (Node, Option<Node>) = {
            let prev_batch = self.h.shape().batch();
            if prev_batch > 1 {
                let batch = x.shape().batch();
                if batch > prev_batch {
                    panic!(
                        "batch size must be smaller than or equal to the previous batch size: x.shape: {}, lstm.h.shape: {}",
                        x.shape(),
                        self.h.shape()
                    );
                } else if batch < prev_batch {
                    let h = F::batch::slice(&self.h, 0, batch);
                    let h_rest = F::batch::slice(&self.h, batch, prev_batch);
                    (F::concat(&vec![x, &h], 0), Some(h_rest))
                } else {
                    (F::concat(&vec![x, &self.h], 0), None)
                }
            } else {
                (F::concat(&vec![x, &self.h], 0), None)
            }
        };
        let u = F::matmul(&self.w, lstm_in) + &self.b;
        let v = F::split(u, 0, 4);
        let i = F::sigmoid(&v[0]);
        let f = F::sigmoid(&v[1]);
        let o = F::sigmoid(&v[2]);
        let j = F::tanh(&v[3]);

        match h_rest {
            Some(h) => {
                let batch = x.shape().batch();
                let prev_batch = self.c.shape().batch();
                let c = F::batch::slice(&self.c, 0, batch);
                let c_rest = F::batch::slice(&self.c, batch, prev_batch);
                let c = i * j + f * c;
                let y = o * F::tanh(&c);
                self.h = F::batch::concat(&vec![&y, &h]);
                self.c = F::batch::concat(&vec![c, c_rest]);
                y
            }
            None => {
                self.c = i * j + f * &self.c;
                let y = o * F::tanh(&self.c);
                self.h = y.clone();
                y
            }
        }
    }

    pub fn initialized(&self) -> bool {
        self.pw.valid()
    }

    pub fn ready(&self) -> bool {
        self.w.valid()
    }

    pub fn get_c(&self) -> &Node {
        &self.c
    }

    pub fn get_h(&self) -> &Node {
        &self.h
    }

    pub fn input_size(&self) -> u32 {
        let s = self.pw.shape();
        s.at(1) - s.at(0) / 4
    }

    pub fn output_size(&self) -> u32 {
        self.pw.shape().at(0) / 4
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BiLSTM {
    lstms: Vec<(LSTM, LSTM)>,
    dropout_rate: f32,
}

impl BiLSTM {
    pub fn new(n_layers: usize, dropout: f32) -> Self {
        assert!(n_layers > 0, "`n_layers` must be greater than 0");
        let lstms = (0..n_layers).map(|_| (LSTM::new(), LSTM::new())).collect();
        BiLSTM {
            lstms: lstms,
            dropout_rate: dropout,
        }
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        let mut iter = self.lstms.iter_mut();
        {
            let &mut (ref mut lstm_f, ref mut lstm_b) = iter.next().unwrap();
            lstm_f.init(in_size, out_size);
            lstm_b.init(in_size, out_size);
        }
        for &mut (ref mut lstm_f, ref mut lstm_b) in iter {
            lstm_f.init(out_size * 2, out_size);
            lstm_b.init(out_size * 2, out_size);
        }
    }

    /// Initializes internal values.
    fn reset(&mut self, init_states: Option<Vec<(Option<Node>, Option<Node>)>>, batch_size: u32) {
        let num_layers = self.lstms.len() * 2;
        let out_size = self.output_size() / 2;
        let mut states = init_states.unwrap_or(vec![]);
        debug_assert!(states.len() <= num_layers);
        for _ in 0..(num_layers - states.len()) {
            states.push((
                Some(F::zeros(Shape::from_dims(&[out_size], batch_size))),
                Some(F::zeros(Shape::from_dims(&[out_size], batch_size))),
            ));
        }
        for (i, (c, h)) in states.into_iter().enumerate() {
            let (ref mut lstm_f, ref mut lstm_b) = self.lstms[i / 2];
            if i % 2 == 0 {
                lstm_f.reset(c, h);
            } else {
                if let Some(ref init_c) = c {
                    debug_assert!(init_c.shape().batch() == batch_size);
                }
                if let Some(ref init_h) = h {
                    debug_assert!(init_h.shape().batch() == batch_size);
                }
                lstm_b.reset(c, h);
            }
        }
    }

    pub fn forward(
        &mut self,
        xs: impl AsRef<[Node]>,
        init_states: Option<Vec<(Option<Node>, Option<Node>)>>,
        train: bool,
    ) -> Vec<Node> {
        let xs = xs.as_ref();
        self.reset(init_states, xs[0].shape().batch());
        debug_assert!(self.initialized() && self.ready());
        let mut iter = self.lstms.iter_mut();
        let hs = {
            let &mut (ref mut lstm_f, ref mut lstm_b) = iter.next().unwrap();
            let xs_f = xs.iter();
            let xs_b = xs.iter().rev();
            let hs_f = xs_f.map(|x| lstm_f.forward(x));
            let hs_b = xs_b.map(|x| lstm_b.forward(x));
            hs_f.zip(hs_b.collect::<Vec<Node>>().into_iter().rev())
                .map(|(h_f, h_b)| F::concat(&[h_f, h_b], 0))
                .collect()
        };
        let mut xs_next: Vec<Node> = hs;
        let dropout_rate = self.dropout_rate;
        for &mut (ref mut lstm_f, ref mut lstm_b) in iter {
            let hs = {
                let xs_f = xs_next.iter();
                let xs_b = xs_next.iter().rev();
                let hs_f = xs_f.map(|x| lstm_f.forward(F::dropout(x, dropout_rate, train)));
                let hs_b = xs_b.map(|x| lstm_b.forward(F::dropout(x, dropout_rate, train)));
                hs_f.zip(hs_b.collect::<Vec<Node>>().into_iter().rev())
                    .map(|(h_f, h_b)| F::concat(&[h_f, h_b], 0))
                    .collect()
            };
            xs_next = hs;
        }
        xs_next
    }

    pub fn initialized(&self) -> bool {
        self.lstms[0].0.initialized()
    }

    pub fn ready(&self) -> bool {
        self.lstms[0].0.ready()
    }

    pub fn get_c(&self) -> (&Node, &Node) {
        let (lstm_f, lstm_b) = self.lstms.last().unwrap();
        (lstm_f.get_c(), lstm_b.get_c())
    }

    pub fn get_h(&self) -> (&Node, &Node) {
        let (lstm_f, lstm_b) = self.lstms.last().unwrap();
        (lstm_f.get_h(), lstm_b.get_h())
    }

    pub fn input_size(&self) -> u32 {
        self.lstms[0].0.input_size()
    }

    pub fn output_size(&self) -> u32 {
        self.lstms[0].0.output_size() * 2
    }
}

impl Model for BiLSTM {
    fn register_parameters(&mut self) {
        let handle: *mut _ = self;
        unsafe {
            let model = &mut *handle;
            for (i, layer) in self.lstms.iter_mut().enumerate() {
                layer.0.register_parameters();
                model.add_submodel(&format!("{}.f_lstm", i), &mut layer.0);
                layer.1.register_parameters();
                model.add_submodel(&format!("{}.b_lstm", i), &mut layer.1);
            }
        }
    }

    fn identifier(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        hasher.write(format!("{}-{:p}", "BiLSTM", self).as_bytes());
        hasher.finish()
    }
}

pub fn transpose_sequence(xs: Vec<Node>) -> Vec<Node> {
    let batch_size = xs[0].shape().batch() as usize;
    let mut lengths = vec![xs.len(); batch_size];
    let mut end = batch_size;
    let xs: Vec<Node> = xs
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            let mut s = x.shape();
            let len = s.batch() as usize;
            if len < end {
                for l in lengths[len..end].iter_mut() {
                    *l = i;
                }
                end = len;
            }
            let diff = batch_size - len;
            if diff > 0 {
                s.update_batch(diff as u32);
                F::batch::concat(&[x, F::zeros(s)])
            } else {
                x
            }
        })
        .collect();
    let xs_transposed = F::batch::split(F::transpose(F::concat(&xs, 1)), batch_size as u32);
    xs_transposed
        .into_iter()
        .zip(lengths.into_iter())
        .map(|(x, len)| F::slice(x, 0, 0, len as u32))
        .collect()
}
