use super::Network;

impl Network {
    pub fn forward(&mut self, inputs: &[f64]) {       
        fn dot_multiply(input: &[f64], factors: &[f64]) -> f64 {
            input.iter().zip(factors).map(|(&input, &factor)| input * factor).sum()
        }

        let activations = &self.hyper_params.activations;

        for (((output, net_input), weights), bias) in self.outputs[0].iter_mut()
            .zip(self.net_inputs[0].iter_mut())
            .zip(self.weights[0].iter())
            .zip(self.biases[0].iter()) 
        {
            *net_input = dot_multiply(inputs, weights) + bias;
            *output = (activations[0].function)(*net_input);
        }

        for layer in 1..self.outputs.len() {
            for (neuron, ((net_input, weights), bias)) in self.net_inputs[layer].iter_mut()
                .zip(self.weights[layer].iter())
                .zip(self.biases[layer].iter())
                .enumerate() 
            {
                *net_input = dot_multiply(&self.outputs[layer - 1], weights) + bias;
                self.outputs[layer][neuron] = (activations[layer].function)(*net_input);
            }
        }    
    }
}
