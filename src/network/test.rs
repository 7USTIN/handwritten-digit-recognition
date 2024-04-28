use crate::dataset::Data;
use super::state::Network;

impl Network {    
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

            if Data::one_hot_encode(predicted_output_index) == *target {
                correct_count += 1.0
            }

            for (output, target) in self.outputs.last().unwrap().iter().zip(target) {
                cost += 0.5 * (*target - *output).powi(2);
            }
        }

        let accuracy = correct_count / data.targets.len() as f64;
        let cost = cost / data.inputs.len() as f64;        

        (accuracy, cost)
    }
}
