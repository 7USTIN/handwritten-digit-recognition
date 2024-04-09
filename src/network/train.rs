use crate::monitor::monitor_training;
use crate::dataset::Data;
use super::state::{ Network, Regularization, LearningRate, DecayMethod, HyperParams };

use std::time::{ Instant, Duration };
use rand::{ thread_rng, Rng };

impl Network {
    fn generate_dropout_mask(&mut self) {
        let HyperParams { regularization, .. } = &self.hyper_params;
        let Regularization { dropout_rate, .. } = regularization;
        
        let mut rng = thread_rng();

        for dropout in self.dropout_mask[0].iter_mut() {
            *dropout = rng.gen_bool(1.0 - dropout_rate.input_layer) as u16 as f64;
        }

        for layer in 1..self.dropout_mask.len() - 1 {
            for neuron in 0..self.dropout_mask[layer].len() {
                self.dropout_mask[layer][neuron] = rng.gen_bool(1.0 - dropout_rate.hidden_layer) as u16 as f64;
            }
        }
    }

    fn set_all_active_dropout_mask(&mut self) {
        for layer in 0..self.dropout_mask.len() - 1 {
            for neuron in 0..self.dropout_mask[layer].len() {
                self.dropout_mask[layer][neuron] = 1.0;
            }
        }
    }

    fn inverse_dropout(&mut self) {
        let HyperParams { regularization, .. } = &self.hyper_params;
        let Regularization { dropout_rate, .. } = regularization;
        
        for (layer, weights) in self.weights.iter_mut().enumerate() {
            let factor = match layer {
                0 => 1.0 / (1.0 - dropout_rate.input_layer),
                _ => 1.0 / (1.0 - dropout_rate.hidden_layer),
            };

            for weights in weights.iter_mut() {
                for weight in weights.iter_mut() {
                    *weight *= factor;
                }
            }

            for bias in self.biases[layer].iter_mut() {
                *bias *= factor;
            }
        }
    }
    
    fn batch_update(&mut self, chunk_size: f64) {
        for (weights, weight_updates) in self.weights.iter_mut()
            .zip(self.batch.weight_updates.iter_mut())
        {
            for (weights, weight_updates) in weights.iter_mut()
                .zip(weight_updates.iter_mut())
            {
                for (weight, weight_update) in weights.iter_mut()
                    .zip(weight_updates.iter_mut())
                {
                    *weight = *weight_update / chunk_size;
                    *weight_update = 0.0;
                }
            }
        }
        
        for (biases, bias_updates) in self.biases.iter_mut()
            .zip(self.batch.bias_updates.iter_mut())
        {
            for (bias, bias_update) in biases.iter_mut()
                .zip(bias_updates.iter_mut()) 
            {
                *bias = *bias_update / chunk_size;
                *bias_update = 0.0;
            }
        }
    }

    fn learning_rate_decay(&mut self, epoch: u32) {
        let LearningRate { alpha, decay, restart } = &mut self.hyper_params.learning_rate;

        let adjusted_epoch;

        if let Some(restart) = &restart {
            match (epoch + 1) % restart.interval == 0 {
                true => {
                    *alpha = restart.alpha;
                    return;
                },
                false => ()
            }   

            adjusted_epoch = (epoch + 1) % restart.interval;
        } else {
            adjusted_epoch = epoch + 1;
        }
        
        if let Some(decay) = &decay {
            match decay.method {
                DecayMethod::Step(decay_step) => {
                    match adjusted_epoch % decay_step == 0 {
                        true => *alpha *= decay.rate,
                        false => ()
                    }
                },
                DecayMethod::Exponential => *alpha *= decay.rate.powi(adjusted_epoch as i32),
                DecayMethod::Inverse => *alpha /= 1.0 + decay.rate * adjusted_epoch as f64,
            }            
        }
    }

    pub fn train(&mut self, training_data: &Data, testing_data: &Data, epochs: u32) {       
        let mut duration = Duration::ZERO;
        
        for epoch in 0..epochs {          
            let timestamp = Instant::now();
        
            for (inputs, targets) in training_data.inputs.chunks(self.hyper_params.batch_size)
                .zip(training_data.targets.chunks(self.hyper_params.batch_size))
            {                
                self.generate_dropout_mask();
                
                for (inputs, targets) in inputs.iter()
                    .zip(targets.iter())
                {                
                    self.optimizer.iteration += 1;

                    self.forward(inputs);
                    self.backward(inputs, targets);  
                }                
                
                self.batch_update(inputs.len() as f64);
            }
            
            duration += timestamp.elapsed();

            monitor_training(
                epochs, epoch, self.hyper_params.learning_rate.alpha, self.test(testing_data), duration
            );

            self.learning_rate_decay(epoch);
        }

        self.set_all_active_dropout_mask();
        self.inverse_dropout();
    }

    pub fn test(&mut self, data: &Data) -> (f64, f64) {
        let mut correct_count = 0.0;
        let mut cost = 0.0;

        for (input, target) in data.inputs.iter().zip(data.targets.iter()) {
            self.forward(input);

            let predicted_output_index = self.outputs.last().unwrap()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            match Data::one_hot_encode(predicted_output_index) == *target {
                true => correct_count += 1.0,
                false => ()
            }

            for (output, target) in self.outputs.last().unwrap().iter().zip(target) {
                cost += 0.5 * (*target - *output).powi(2);
            }
        }

        let accuracy = (correct_count / data.targets.len() as f64) * 100.0;
        let cost = cost / data.inputs.len() as f64;        

        (accuracy, cost)
    }
}
