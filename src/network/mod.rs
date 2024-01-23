use super::monitor::monitor_training;
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
    pub fn train(&mut self, training_data: &Data, testing_data: &Data, epochs: u32) {       
        let mut costs = self.outputs.clone();
        
        for epoch in 0..epochs {        
            for index in 0..training_data.inputs.len() {                
                self.forward(&training_data.inputs[index]);
                self.backward(&training_data.inputs[index], &training_data.targets[index], &mut costs);  
            }

            monitor_training(epoch, self.test(testing_data));            
        }
    }

    pub fn test(&mut self, data: &Data) -> f64 {
        let mut correct_count = 0.0;

        for (input, target) in data.inputs.iter().zip(data.targets.iter()) {
            self.forward(input);

            if target == self.outputs.last().unwrap() {
                correct_count += 1.0;
            }
        }

        (correct_count / data.targets.len() as f64) * 100.0
    }
}
