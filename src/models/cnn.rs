use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;
use primitiv::initializers as I;
use primitiv::node_functions as F;

#[derive(Debug)]
pub struct Conv2D {
    model: Model,
    pw: Parameter,
    pub padding: (u32, u32),
    pub stride: (u32, u32),
    pub dilation: (u32, u32),
}

impl Conv2D {
    pub fn new(padding: (u32, u32), stride: (u32, u32), dilation: (u32, u32)) -> Self {
        let mut m = Conv2D {
            model: Model::new(),
            pw: Parameter::new(),
            padding: padding,
            stride: stride,
            dilation: dilation,
        };
        m.model.add_parameter("w", &mut m.pw);
        m
    }

    /// Initializes the model.
    pub fn init(&mut self, in_channels: u32, out_channels: u32, kernel: (u32, u32)) {
        self.pw.init_by_initializer(
            [kernel.0, kernel.1, in_channels, out_channels],
            &I::XavierUniformConv2D::new(1.0),
        );
    }

    /// Forwarding.
    pub fn forward<N: AsRef<Node>>(&mut self, x: N) -> Node {
        F::conv2d(
            x.as_ref(),
            F::parameter(&mut self.pw),
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            self.dilation.0,
            self.dilation.1,
        )
    }

    pub fn initialized(&self) -> bool {
        self.pw.valid()
    }
}

impl_model!(Conv2D, model);

impl Default for Conv2D {
    fn default() -> Self {
        Conv2D::new((0, 0), (1, 1), (1, 1))
    }
}
