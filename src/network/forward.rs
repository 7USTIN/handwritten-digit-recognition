use super::state::Network;

impl Network {
    fn dot_product(factors_1: &[f64], factors_2: &[f64]) -> f64 {
        factors_1.iter().zip(factors_2)
            .map(|(&factor_1, &factor_2)| factor_1 * factor_2)
            .sum()
    }

    pub fn forward(&mut self, inputs: &[f64]) {       
        let activations = &self.hyper_params.activations;

        for ((((output, net_input), weights), bias), dropout_mask) in self.outputs[0].iter_mut()
            .zip(self.net_inputs[0].iter_mut())
            .zip(self.weights[0].iter())
            .zip(self.biases[0].iter()) 
            .zip(self.dropout_mask[0].iter()) 
        {
            *net_input = Self::dot_product(inputs, weights) * dropout_mask + bias;
            *output = (activations[0].function)(*net_input);
        }

        for layer in 1..self.outputs.len() {
            for (neuron, (((net_input, weights), bias), dropout_mask)) in self.net_inputs[layer].iter_mut()
                .zip(self.weights[layer].iter())
                .zip(self.biases[layer].iter())
                .zip(self.dropout_mask[layer].iter())
                .enumerate() 
            {
                *net_input = Self::dot_product(&self.outputs[layer - 1], weights) * dropout_mask + bias;
                self.outputs[layer][neuron] = (activations[layer].function)(*net_input);
            }
        }    
    }
}
