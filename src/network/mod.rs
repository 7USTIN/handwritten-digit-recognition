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
    pub fn train(&mut self, data: &Data, epochs: u32) {       
        let mut costs = self.outputs.clone();

        let timestamp = Instant::now();
        
        for epoch in 0..epochs {
            for i in 0..data.inputs.len() {
                if i % 10000 == 0 {
                    println!("{epoch}, {i}: {:?}", timestamp.elapsed());
                }
                
                self.forward(&data.inputs[i]);
                self.backward(&data.inputs[i], &data.targets[i], &mut costs);  
            }
        }

        println!("{:?}", timestamp.elapsed());
    }

    pub fn test(&mut self, data: &Data) {
        let timestamp = Instant::now();

        let mut correct_count = 0.0;

        for (input, target) in data.inputs.iter().zip(data.targets.iter()) {
            self.forward(input);

            if target == self.outputs.last().unwrap() {
                correct_count += 1.0;
            }
        }

        let accuaracy = (correct_count / data.targets.len() as f64) * 100.0;

        println!("Accuaracy: {:.3}%", accuaracy);
          
        println!("{:?}", timestamp.elapsed());
    }
}
