use primitiv::functions as F;
use primitiv::initializers::XavierUniformConv2D;
use primitiv::Initializer;
use primitiv::Parameter;
use primitiv::Variable;

#[derive(Debug, Model, Serialize, Deserialize)]
pub struct Conv2D {
    pw: Parameter,
    pub padding: (u32, u32),
    pub stride: (u32, u32),
    pub dilation: (u32, u32),
}

impl Conv2D {
    pub fn new(padding: (u32, u32), stride: (u32, u32), dilation: (u32, u32)) -> Self {
        Conv2D {
            pw: Parameter::new(),
            padding,
            stride,
            dilation,
        }
    }

    pub fn init(&mut self, in_channels: u32, out_channels: u32, kernel: (u32, u32)) {
        self.init_by_initializer(
            in_channels,
            out_channels,
            kernel,
            &Self::default_initializer(),
        );
    }

    pub fn init_by_initializer<I: Initializer>(
        &mut self,
        in_channels: u32,
        out_channels: u32,
        kernel: (u32, u32),
        initializer: &I,
    ) {
        self.pw
            .init_by_initializer([kernel.0, kernel.1, in_channels, out_channels], initializer);
    }

    pub fn forward<V: Variable, X: AsRef<V>>(&mut self, x: X) -> V {
        F::conv2d(
            x.as_ref(),
            F::parameter::<V>(&mut self.pw),
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            self.dilation.0,
            self.dilation.1,
        )
    }

    pub fn default_initializer() -> impl Initializer {
        XavierUniformConv2D::default()
    }
}

impl Default for Conv2D {
    fn default() -> Self {
        Conv2D::new((0, 0), (1, 1), (1, 1))
    }
}
