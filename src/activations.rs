pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

impl Activation {
    pub fn get(names: &[&str]) -> Vec<Self> {
        names.iter().map(|&name| {
            match name {
                "BINARY_STEP" => BINARY_STEP,
                "RELU" => RELU,
                "LEAKY_RELU_01" => LEAKY_RELU_01,
                "LEAKY_RELU_001" => LEAKY_RELU_001,
                "ELU_1" => ELU_1,
                "GELU" => GELU,
                "SIGMOID" => SIGMOID,
                "SWISH" => SWISH,
                "TANH" => TANH,
                _ => panic!("Invalid activation function!")
            }
        }).collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f64) -> f64 {
    let exp_pos = x.exp();
    let exp_neg = (-x).exp();

    (exp_pos - exp_neg) / (exp_pos + exp_neg)
}

const BINARY_STEP: Activation = Activation {
    function: |x| f64::from(u8::from(x > 0.0)),
    derivative: |_| 1.0,
};

const SIGMOID: Activation = Activation {
    function: sigmoid,
    derivative: |x| sigmoid(x) * (1.0 - sigmoid(x)),
};

const SWISH: Activation = Activation {
    function: |x| x * sigmoid(x),
    derivative: |x| x.mul_add(sigmoid(x), sigmoid(x) * (1.0 - sigmoid(x))),
};

const TANH: Activation = Activation {
    function: tanh,
    derivative: |x| tanh(x).mul_add(-tanh(x), 1.0),
};

const RELU: Activation = Activation {
    function: |x| x.max(0.0),
    derivative: |x| f64::from(u8::from(x > 0.0)),
};

const LEAKY_RELU_01: Activation = Activation {
    function: |x| x.max(0.1 * x),
    derivative: |x| if x > 0.0 { 1.0 } else { 0.1 },
};

const LEAKY_RELU_001: Activation = Activation {
    function: |x| x.max(0.01 * x),
    derivative: |x| if x > 0.0 { 1.0 } else { 0.01 },
};

const ELU_1: Activation = Activation {
    function: |x| if x > 0.0 { x } else { 1.0 * x.exp_m1() },
    derivative: |x| if x > 0.0 { 1.0 } else { 1.0_f64.mul_add(x.exp_m1(), 1.0) },
};

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

const GELU: Activation = Activation {
    function: |x| 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * 0.044_715_f64.mul_add(x.powi(3), x))),
    derivative: |x| {
        let intermedia_calculation = SQRT_2_OVER_PI * 0.044_715_f64.mul_add(x.powi(3), x);            

        0.5_f64.mul_add(1.0 + tanh(intermedia_calculation), 0.5 * 0.65 * (1.0 / x.cosh()).powi(2)
        * intermedia_calculation
        * SQRT_2_OVER_PI * (3.0 * 0.044_715_f64).mul_add(x.powi(2), 1.0))
    }
};
