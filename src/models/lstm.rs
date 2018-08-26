use primitiv::functions as F;
use primitiv::initializers::Uniform;
use primitiv::Initializer;
use primitiv::Model;
use primitiv::Parameter;
use primitiv::Variable;

/// Hand-written LSTM with input/forget/output gates and no peepholes.
/// Formulation:
///   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
///   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
///   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
///   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
///   c[t] = i * j + f * c[t-1]
///   h[t] = o * tanh(c[t])
#[derive(Debug, Model, Serialize, Deserialize)]
pub struct LSTMCell<V: Variable> {
    pw: Parameter,
    pb: Parameter,
    w: V,
    b: V,
    h: V,
    c: V,
}

impl<V: Variable> LSTMCell<V> {
    pub fn new() -> Self {
        LSTMCell {
            pw: Parameter::new(),
            pb: Parameter::new(),
            w: V::default(),
            b: V::default(),
            h: V::default(),
            c: V::default(),
        }
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.init_by_initializer(in_size, out_size, &Self::default_initializer());
    }

    /// Initializes the model by a initializer.
    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        in_size: u32,
        out_size: u32,
        initializer: &I,
    ) {
        self.pw
            .init_by_initializer([4 * out_size, in_size + out_size], initializer);
        let mut bias_values = vec![0.0; 4 * out_size as usize];
        for i in (out_size as usize)..(2 * out_size as usize) {
            bias_values[i] = 1.0;
        }
        self.pb.init_by_values([4 * out_size], &bias_values);
    }

    /// Initializes internal values.
    pub fn reset(&mut self, init_c: Option<V>, init_h: Option<V>) {
        let out_size = self.pw.shape().at(0) / 4;
        self.w = F::parameter(&mut self.pw);
        self.b = F::parameter(&mut self.pb);
        self.c = match init_c {
            Some(v) => {
                assert!(v.valid(), "init_c must be valid");
                v
            }
            None => F::zeros([out_size]),
        };
        self.h = match init_h {
            Some(v) => {
                assert!(v.valid(), "init_h must be valid");
                v
            }
            None => F::zeros([out_size]),
        };
    }

    /// One step forwarding.
    pub fn forward<X: AsRef<V>>(&mut self, x: X) -> V {
        debug_assert!(self.w.valid(), "call `reset` before forward");
        let x = x.as_ref();
        let (lstm_in, h_rest): (V, Option<V>) = {
            let prev_batch = self.h.shape().batch();
            if prev_batch > 1 {
                let batch = x.shape().batch();
                if batch > prev_batch {
                    panic!(
                        "batch size must be smaller than or equal to the previous batch size: \
                         x.shape: {}, lstm.h.shape: {}",
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

    pub fn cell_state(&self) -> &V {
        &self.c
    }

    pub fn hidden_state(&self) -> &V {
        &self.h
    }

    pub fn input_size(&self) -> u32 {
        let s = self.pw.shape();
        s.at(1) - s.at(0) / 4
    }

    pub fn output_size(&self) -> u32 {
        self.pw.shape().at(0) / 4
    }

    pub fn default_initializer() -> impl Initializer {
        Uniform::new(-0.1, 0.1)
    }
}

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct LSTM<V: Variable> {
    #[primitiv(submodel)]
    lstm_cells: Vec<LSTMCell<V>>,
    dropout_rate: f32,
}

impl<V: Variable> LSTM<V> {
    pub fn new(n_layers: usize, dropout_rate: f32) -> Self {
        assert!(n_layers > 0, "`n_layers` must be greater than 0");
        LSTM {
            lstm_cells: (0..n_layers).map(|_| LSTMCell::new()).collect(),
            dropout_rate,
        }
    }

    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.init_by_initializer(in_size, out_size, &Self::default_initializer());
    }

    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        in_size: u32,
        out_size: u32,
        initializer: &I,
    ) {
        let mut iter = self.lstm_cells.iter_mut();
        {
            let lstm_cell = iter.next().unwrap();
            lstm_cell.init_by_initializer(in_size, out_size, initializer);
        }
        for lstm_cell in iter {
            lstm_cell.init_by_initializer(out_size, out_size, initializer);
        }
    }

    fn reset(&mut self, init_states: Option<Vec<(Option<V>, Option<V>)>>, batch_size: u32) {
        let num_cells = self.lstm_cells.len();
        let out_size = self.output_size();
        let mut init_states = init_states.unwrap_or(Vec::with_capacity(num_cells));
        debug_assert!(init_states.len() <= num_cells);
        for _ in 0..(num_cells - init_states.len()) {
            init_states.push((
                Some(F::zeros(([out_size], batch_size))),
                Some(F::zeros(([out_size], batch_size))),
            ));
        }
        for (lstm_cell, (c, h)) in self.lstm_cells.iter_mut().zip(init_states) {
            lstm_cell.reset(c, h)
        }
    }

    pub fn forward<X: AsRef<[V]>>(
        &mut self,
        xs: X,
        init_states: Option<Vec<(Option<V>, Option<V>)>>,
        train: bool,
    ) -> Vec<V> {
        let xs = xs.as_ref();
        let batch_size = xs[0].shape().batch().max(xs[xs.len() - 1].shape().batch());
        self.reset(init_states, batch_size);
        let mut iter = self.lstm_cells.iter_mut();
        let hs = {
            let lstm_cell = iter.next().unwrap();
            xs.iter().map(|x| lstm_cell.forward(x)).collect()
        };
        let mut xs_next: Vec<V> = hs;
        let dropout_rate = self.dropout_rate;
        for lstm_cell in iter {
            xs_next = xs_next
                .iter()
                .map(|x| lstm_cell.forward(F::dropout(x, dropout_rate, train)))
                .collect();
        }
        xs_next
    }

    pub fn cell_state(&self) -> &V {
        self.lstm_cells.last().unwrap().cell_state()
    }

    pub fn hidden_state(&self) -> &V {
        self.lstm_cells.last().unwrap().hidden_state()
    }

    pub fn input_size(&self) -> u32 {
        self.lstm_cells[0].input_size()
    }

    pub fn output_size(&self) -> u32 {
        self.lstm_cells[0].output_size()
    }

    pub fn num_layers(&self) -> usize {
        self.lstm_cells.len()
    }

    pub fn default_initializer() -> impl Initializer {
        LSTMCell::<V>::default_initializer()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BiLSTM<V: Variable> {
    lstm_cells: Vec<(LSTMCell<V>, LSTMCell<V>)>,
    dropout_rate: f32,
}

impl<V: Variable> BiLSTM<V> {
    pub fn new(n_layers: usize, dropout_rate: f32) -> Self {
        assert!(n_layers > 0, "`n_layers` must be greater than 0");
        BiLSTM {
            lstm_cells: (0..n_layers)
                .map(|_| (LSTMCell::new(), LSTMCell::new()))
                .collect(),
            dropout_rate,
        }
    }

    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.init_by_initializer(in_size, out_size, &Self::default_initializer());
    }

    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        in_size: u32,
        out_size: u32,
        initializer: &I,
    ) {
        let mut iter = self.lstm_cells.iter_mut();
        {
            let (lstm_cell_f, lstm_cell_b) = iter.next().unwrap();
            lstm_cell_f.init_by_initializer(in_size, out_size, initializer);
            lstm_cell_b.init_by_initializer(in_size, out_size, initializer);
        }
        for (lstm_cell_f, lstm_cell_b) in iter {
            lstm_cell_f.init_by_initializer(out_size * 2, out_size, initializer);
            lstm_cell_b.init_by_initializer(out_size * 2, out_size, initializer);
        }
    }

    fn reset(&mut self, init_states: Option<Vec<(Option<V>, Option<V>)>>, batch_size: u32) {
        let num_cells = self.lstm_cells.len() * 2;
        let out_size = self.output_size() / 2;
        let mut init_states = init_states.unwrap_or(Vec::with_capacity(num_cells));
        debug_assert!(init_states.len() <= num_cells);
        for _ in 0..(num_cells - init_states.len()) {
            init_states.push((
                Some(F::zeros(([out_size], batch_size))),
                Some(F::zeros(([out_size], batch_size))),
            ));
        }
        for (i, (c, h)) in init_states.into_iter().enumerate() {
            let (lstm_cell_f, lstm_cell_b) = &mut self.lstm_cells[i / 2];
            if i % 2 == 0 {
                lstm_cell_f.reset(c, h);
            } else {
                if cfg!(debug_assertion) {
                    if let Some(ref c) = c {
                        debug_assert!(c.shape().batch() == batch_size);
                    }
                    if let Some(ref h) = h {
                        debug_assert!(h.shape().batch() == batch_size);
                    }
                }
                lstm_cell_b.reset(c, h);
            }
        }
    }

    pub fn forward<X: AsRef<[V]>>(
        &mut self,
        xs: X,
        init_states: Option<Vec<(Option<V>, Option<V>)>>,
        train: bool,
    ) -> Vec<V> {
        let xs = xs.as_ref();
        self.reset(init_states, xs[0].shape().batch());
        let mut iter = self.lstm_cells.iter_mut();
        let hs = {
            let (lstm_cell_f, lstm_cell_b) = iter.next().unwrap();
            let xs_f = xs.iter();
            let xs_b = xs.iter().rev();
            let hs_f = xs_f.map(|x| lstm_cell_f.forward(x));
            let hs_b = xs_b.map(|x| lstm_cell_b.forward(x));
            hs_f.zip(hs_b.collect::<Vec<V>>().into_iter().rev())
                .map(|(h_f, h_b)| F::concat(&[h_f, h_b], 0))
                .collect()
        };
        let mut xs_next: Vec<V> = hs;
        let dropout_rate = self.dropout_rate;
        for (lstm_cell_f, lstm_cell_b) in iter {
            let hs = {
                let xs_f = xs_next.iter();
                let xs_b = xs_next.iter().rev();
                let hs_f = xs_f.map(|x| lstm_cell_f.forward(F::dropout(x, dropout_rate, train)));
                let hs_b = xs_b.map(|x| lstm_cell_b.forward(F::dropout(x, dropout_rate, train)));
                hs_f.zip(hs_b.collect::<Vec<V>>().into_iter().rev())
                    .map(|(h_f, h_b)| F::concat(&[h_f, h_b], 0))
                    .collect()
            };
            xs_next = hs;
        }
        xs_next
    }

    pub fn cell_state(&self) -> (&V, &V) {
        let (lstm_cell_f, lstm_cell_b) = self.lstm_cells.last().unwrap();
        (lstm_cell_f.cell_state(), lstm_cell_b.cell_state())
    }

    pub fn hidden_state(&self) -> (&V, &V) {
        let (lstm_cell_f, lstm_cell_b) = self.lstm_cells.last().unwrap();
        (lstm_cell_f.hidden_state(), lstm_cell_b.hidden_state())
    }

    pub fn input_size(&self) -> u32 {
        self.lstm_cells[0].0.input_size()
    }

    pub fn output_size(&self) -> u32 {
        self.lstm_cells[0].0.output_size() * 2
    }

    pub fn num_layers(&self) -> usize {
        self.lstm_cells.len()
    }

    pub fn default_initializer() -> impl Initializer {
        LSTMCell::<V>::default_initializer()
    }
}

impl<V: Variable> Model for BiLSTM<V> {
    fn register_parameters(&mut self) {
        let handle: *mut _ = self;
        unsafe {
            let model = &mut *handle;
            for (i, (lstm_cell_f, lstm_cell_b)) in self.lstm_cells.iter_mut().enumerate() {
                lstm_cell_f.register_parameters();
                model.add_submodel(&format!("lstm_cells.{}.f", i), lstm_cell_f);
                lstm_cell_b.register_parameters();
                model.add_submodel(&format!("lstm_cells.{}.b", i), lstm_cell_b);
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
