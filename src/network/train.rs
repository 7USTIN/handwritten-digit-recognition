use crate::monitor::monitor_training;
use crate::dataset::Data;
use super::state::Network;

impl Network {
    pub fn train(&mut self, training_data: &Data, testing_data: &Data, epochs: u32) {       
        for epoch in 0..epochs {        
            for index in 0..training_data.inputs.len() {                
                self.optimizer.iteration += 1;

                self.forward(&training_data.inputs[index]);
                self.backward(&training_data.inputs[index], &training_data.targets[index]);  
            }

            monitor_training(epoch, self.test_accuracy(testing_data));            
        }
    }

    fn test_accuracy(&mut self, data: &Data) -> f64 {
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
