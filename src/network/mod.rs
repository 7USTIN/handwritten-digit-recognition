use std::time::Instant;

use super::activations::Activation;
use super::dataset::Data;

mod initialization;
mod forward;
mod backward;

pub type Vec2D = Vec<Vec<f64>>;

pub struct Regularizer {
    pub l1: f64,
    pub l2: f64
}

pub struct Regularization {
    pub weight: Regularizer,
    pub bias: Regularizer
}

pub struct HyperParams {
    pub composition: Vec<usize>,
    pub activations: Vec<Activation>,
    pub learning_rate: f64,
    pub regularization: Regularization
}

pub struct Network {
    weights: Vec<Vec2D>,
    biases: Vec2D,
    net_inputs: Vec2D,
    outputs: Vec2D,
    hyper_params: HyperParams,
}

impl Network {
    pub fn train(&mut self, data: Data, epochs: u32) {       
        let mut costs = self.outputs.clone();

        let timestamp = Instant::now();
        
        for _ in 0..epochs {
            for i in 0..data.inputs.len() {
                self.forward(&data.inputs[i]);
                self.backward(&data.inputs[i], &data.targets[i], &mut costs);  
            }
        }

        println!("{:?}", timestamp.elapsed());
    }

    pub fn test(&mut self, data: Data) {
        let timestamp = Instant::now();

        for input in data.inputs {
            self.forward(&input);

            println!("{:?}", self.outputs.last().unwrap());
        }
          
        println!("{:?}", timestamp.elapsed());
    }
}
