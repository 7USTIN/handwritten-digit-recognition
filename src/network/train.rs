use super::{ 
    optimizations::{ regularization::Dropout, batch::Batch, early_stopping::EarlyStopping, learning_rate::LearningRate }, 
    state::Network 
};
use crate::monitor::monitor_training;
use crate::dataset::Data;

use std::time::{ Instant, Duration };

impl Network {
    pub fn train(&mut self, train_data: &Data, validation_data: &Data) {       
        let mut duration = Duration::ZERO;
        let mut epoch = 0;
        
        loop {
            let timestamp = Instant::now();
            epoch += 1;

            for (inputs, targets) in train_data.inputs.chunks(self.hyper_params.batch_size)
                .zip(train_data.targets.chunks(self.hyper_params.batch_size))
            {                
                Dropout::generate_mask(self);

                for (inputs, targets) in inputs.iter()
                    .zip(targets.iter())
                {                
                    self.optimizer.iteration += 1;

                    self.forward(inputs);
                    self.backward(inputs, targets);             
                }                
                
                Batch::update(self, inputs.len() as f64);
            }

            Dropout::set_all_active_mask(self);
            
            duration += timestamp.elapsed();

            let (accuracy, cost) = self.test(validation_data);
            let early_stop = EarlyStopping::check(self, accuracy);

            monitor_training(
                epoch, self.hyper_params.learning_rate.alpha, accuracy, cost, duration, early_stop
            );

            if early_stop {
                break;
            }
            
            LearningRate::update(self, &epoch);
        }
    }
}
