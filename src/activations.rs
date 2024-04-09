#[allow(dead_code)]

pub enum ActivationType {
    LeakyRelu,
    Elu,
    Gelu,
    Sigmoid,
    Swish,
    Tanh,
}

use ActivationType::*;

pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

impl Activation {
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f64) -> f64 {
        let exp_pos = x.exp();
        let exp_neg = (-x).exp();

        (exp_pos - exp_neg) / (exp_pos + exp_neg)
    }

    pub fn get(activations: &[ActivationType]) -> Vec<Self> {
        activations.iter().map(|activation| {            
            match activation {
                Sigmoid => Activation {
                    function: Self::sigmoid,
                    derivative: |x| {
                        let sigmoid_x = Self::sigmoid(x);
                        
                        sigmoid_x * (1.0 - sigmoid_x)
                    }
                },
                Swish => Activation {
                    function: |x| x * Self::sigmoid(x),
                    derivative: |x| {
                        let sigmoid_x = Self::sigmoid(x);
                    
                        x * sigmoid_x + sigmoid_x * (1.0 - sigmoid_x)
                    }
                },
                Tanh => Activation {
                    function: Self::tanh,
                    derivative: |x| 1.0 - Self::tanh(x).powi(2),
                },
                LeakyRelu => Activation {
                    function: |x| x.max(0.01 * x),
                    derivative: |x| match x >= 0.0 { true => 1.0, false => 0.01 }
                },
                Elu => Activation {
                    function: |x| match x >= 0.0 { true => x, false => 1.0 * x.exp_m1() },
                    derivative: |x| match x >= 0.0 { true => 1.0, false => 1.0 * x.exp_m1() + 1.0 } 
                },
                Gelu => Activation {               
                    function: |x| 0.5 * x * (1.0 + Self::tanh(SQRT_2_OVER_PI * (0.044_715 * x.powi(3) + x))),
                    derivative: |x| {
                        let sub_calculation = SQRT_2_OVER_PI * (0.044_715 * x.powi(3) + x); 

                        0.5 * (1.0 + Self::tanh(sub_calculation) + 0.5 * x * (1.0 / x.cosh()).powi(2))
                        * sub_calculation
                        * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044_715_f64 * x.powi(2))
                    }
                }
            }
        }).collect()
    }
}
