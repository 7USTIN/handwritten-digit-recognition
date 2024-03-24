use crate::monitor::monitor_training;
use crate::dataset::Data;
use super::state::Network;
use std::time::{ Instant, Duration };

impl Network {
    pub fn train(&mut self, training_data: &Data, testing_data: &Data, epochs: u32) {
        let mut duration = Duration::ZERO;
        
        for epoch in 0..epochs {          
            let timestamp = Instant::now();
        
            for (inputs, targets) in training_data.inputs.chunks(self.hyper_params.batch_size)
                .zip(training_data.targets.chunks(self.hyper_params.batch_size))
            {                
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

            monitor_training(epochs, epoch, self.test(testing_data), duration);
        }
    }

    fn test(&mut self, data: &Data) -> (f64, f64) {
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

            if Data::one_hot_encode(predicted_output_index) == *target {
                correct_count += 1.0;
            }

            for (output, target) in self.outputs.last().unwrap().iter().zip(target) {
                cost += 0.5 * (*target - *output).powi(2);
            }
        }

        let accuracy = (correct_count / data.targets.len() as f64) * 100.0;
        let avg_cost = cost / self.outputs.len() as f64 / data.inputs.len() as f64;        

        (accuracy, avg_cost)
    }
}
