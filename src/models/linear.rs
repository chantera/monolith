use primitiv::functions as F;
use primitiv::initializers::{Constant, XavierUniform};
use primitiv::Initializer;
use primitiv::Parameter;
use primitiv::Variable;

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct Linear {
    pw: Parameter,
    pb: Option<Parameter>,
}

impl Linear {
    pub fn new(use_bias: bool) -> Self {
        Linear {
            pw: Parameter::new(),
            pb: if use_bias {
                Some(Parameter::new())
            } else {
                None
            },
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
        self.pw
            .init_by_initializer([out_size, in_size], initializer);
        if let Some(ref mut pb) = self.pb {
            pb.init_by_initializer([out_size], &Constant::new(0.0));
        }
    }

    pub fn forward<V: Variable, X: AsRef<V>>(&mut self, x: X) -> V {
        let w: V = F::parameter(&mut self.pw);
        let mut y = F::matmul(w, x);
        if let Some(ref mut pb) = self.pb {
            let b: V = F::parameter(pb);
            y = y + b;
        }
        y
    }

    pub fn default_initializer() -> impl Initializer {
        XavierUniform::default()
    }
}
