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
            let mut b: V = F::parameter(pb);
            let s = y.shape();
            if s.dims().len() == 2 {
                b = F::broadcast(b, 1, s.at(1));
            }
            y = y + b;
        }
        y
    }

    pub fn default_initializer() -> impl Initializer {
        XavierUniform::default()
    }
}

impl Default for Linear {
    fn default() -> Self {
        Linear::new(true)
    }
}

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct Bilinear {
    pw: Parameter,
    pu: Option<Parameter>,
    pv: Option<Parameter>,
    pb: Option<Parameter>,
}

impl Bilinear {
    pub fn new(use_bias: (bool, bool, bool)) -> Self {
        Bilinear {
            pw: Parameter::new(),
            pu: if use_bias.0 {
                Some(Parameter::new())
            } else {
                None
            },
            pv: if use_bias.1 {
                Some(Parameter::new())
            } else {
                None
            },
            pb: if use_bias.2 {
                Some(Parameter::new())
            } else {
                None
            },
        }
    }

    pub fn init(&mut self, in_size1: u32, in_size2: u32, out_size: u32) {
        self.init_by_initializer(in_size1, in_size2, out_size, &Self::default_initializer())
    }

    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        in_size1: u32,
        in_size2: u32,
        out_size: u32,
        initializer: &I,
    ) {
        self.pw
            .init_by_initializer([out_size, in_size2, in_size1], initializer);
        if let Some(ref mut pu) = self.pu {
            pu.init_by_initializer([out_size, in_size1], initializer);
        }
        if let Some(ref mut pv) = self.pv {
            pv.init_by_initializer([out_size, in_size2], initializer);
        }
        if let Some(ref mut pb) = self.pb {
            pb.init_by_initializer([out_size], &Constant::new(0.0));
        }
    }

    pub fn forward<V: Variable, X1: AsRef<V>, X2: AsRef<V>>(&mut self, x1: X1, x2: X2) -> V {
        let w: V = F::parameter(&mut self.pw);
        let s = w.shape();
        let out_size = s.at(0);
        let in_size1 = s.at(2);
        let in_size2 = s.at(1);

        // [out, in2, in1] -> [out * in2, in1] -> [out * in2, n1]
        let mut y = F::matmul(F::reshape(w, [out_size * in_size2, in_size1]), x1.as_ref());
        let n1 = {
            let dims = y.shape().dims();
            if dims.len() == 2 {
                y = F::transpose(y);
                Some(dims[1])
            } else {
                debug_assert!(dims.len() <= 1);
                None
            }
        };
        // [n1, out * in2] -> [n1 * out, in2] -> [n1 * out, n2]
        y = F::matmul(
            F::reshape(y, [n1.unwrap_or(1) * out_size, in_size2]),
            x2.as_ref(),
        );
        let n2 = {
            let dims = y.shape().dims();
            if dims.len() == 2 {
                Some(dims[1])
            } else {
                debug_assert!(dims.len() <= 1);
                None
            }
        };
        if let Some(n1) = n1 {
            if let Some(n2) = n2 {
                // [n1, out, n2]
                if out_size > 1 {
                    y = F::reshape(y, [n1, out_size, n2]);
                } else {
                    y = F::reshape(y, [n1, n2]);
                }
            } else {
                // [n1, out]
                if out_size > 1 {
                    y = F::reshape(y, [n1, out_size]);
                } else {
                    y = F::reshape(y, [n1]);
                }
            }
        } else {
            if let Some(n2) = n2 {
                // [out, n2]
                if out_size > 1 {
                    y = F::reshape(y, [out_size, n2]);
                } else {
                    y = F::reshape(y, [n2]);
                }
            } else {
                // [out]
            }
        }

        if let Some(ref mut pu) = self.pu {
            let mut u: V = F::parameter(pu);
            let mut b1 = F::matmul(u, x1); // [out, n1] or [out]
            if n1.is_some() {
                b1 = F::transpose(b1);
                if let Some(n2) = n2 {
                    // [n1, out, n2] + [n1, out]
                    if out_size > 1 {
                        b1 = F::broadcast(b1, 2, n2);
                    } else {
                        b1 = F::broadcast(b1, 1, n2);
                    }
                } else {
                    // [n1, out] + [n1, out]
                }
            } else {
                if let Some(n2) = n2 {
                    // [out, n2] + [out]
                    if out_size > 1 {
                        b1 = F::broadcast(b1, 1, n2);
                    } else {
                        b1 = F::broadcast(b1, 0, n2);
                    }
                } else {
                    // [out] + [out]
                }
            }
            y = y + b1;
        }
        if let Some(ref mut pv) = self.pv {
            let mut v: V = F::parameter(pv);
            let mut b2 = F::matmul(v, x2); // [out, n2] or [out]
            if let Some(n1) = n1 {
                if let Some(n2) = n2 {
                    // [n1, out, n2] + [out, n2]
                    if out_size > 1 {
                        b2 = F::broadcast(F::reshape(b2, [1, out_size, n2]), 0, n1);
                    } else {
                        b2 = F::broadcast(b2, 0, n1);
                    }
                } else {
                    // [n1, out] + [out]
                    b2 = F::broadcast(F::reshape(b2, [1, out_size]), 0, n1);
                }
            } else {
                if let Some(n2) = n2 {
                    // [out, n2] + [out, n2]
                    if out_size > 1 {
                        // pass
                    } else {
                        b2 = F::reshape(b2, [n2]);
                    }
                } else {
                    // [out] + [out]
                }
            }
            y = y + b2;
        }
        if let Some(ref mut pb) = self.pb {
            let mut b: V = F::parameter(pb); // [out]
            if let Some(n1) = n1 {
                if let Some(n2) = n2 {
                    // [n1, out, n2] + [out]
                    if out_size > 1 {
                        b = F::broadcast(F::reshape(b, [1, out_size, 1]), 2, n2);
                    } else {
                        b = F::broadcast(b, 1, n2);
                    }
                    b = F::broadcast(b, 0, n1);
                } else {
                    // [n1, out] + [out]
                    if out_size > 1 {
                        b = F::broadcast(F::reshape(b, [1, out_size]), 0, n1);
                    } else {
                        b = F::broadcast(b, 0, n1);
                    }
                }
            } else {
                if let Some(n2) = n2 {
                    // [out, n2] + [out]
                    if out_size > 1 {
                        b = F::broadcast(b, 1, n2);
                    } else {
                        b = F::broadcast(b, 0, n2);
                    }
                } else {
                    // [out] + [out]
                }
            }
            y = y + b;
        }
        y
    }

    pub fn default_initializer() -> impl Initializer {
        // XavierUniform::default() // TODO(chantera) implement
        Constant::new(0.0)
    }
}

impl Default for Bilinear {
    fn default() -> Self {
        Bilinear::new((true, true, true))
    }
}
