use crate::monitor::monitor_training;
use crate::dataset::Data;
use super::state::Network;
use std::time::{ Instant, Duration };

impl Network {
    pub fn train(&mut self, training_data: &Data, testing_data: &Data, epochs: u32) {
        let mut duration = Duration::ZERO;
        
        for epoch in 0..epochs {          
            let timestamp = Instant::now();
        
            for index in 0..training_data.inputs.len() {                
                self.optimizer.iteration += 1;

                self.forward(&training_data.inputs[index]);
                self.backward(&training_data.inputs[index], &training_data.targets[index]);  
            }
            
            duration += timestamp.elapsed();

            monitor_training(epochs, epoch, self.test(testing_data), duration); 
        }
    }

    fn test(&mut self, data: &Data) -> (f64, f64) {
        let mut cost = 0.0;
        let mut correct_count = 0.0;

        for (input, target) in data.inputs.iter().zip(data.targets.iter()) {
            self.forward(input);

            if target == self.outputs.last().unwrap() {
                correct_count += 1.0;
            }

            for (output, target) in self.outputs.last().unwrap().iter().zip(target) {
                cost += (*output - *target).powi(2);
            }
        }

        let accuracy = (correct_count / data.targets.len() as f64) * 100.0;
        let avg_cost = cost / self.outputs.len() as f64 / data.inputs.len() as f64;        

        (accuracy, avg_cost)
    }
}
