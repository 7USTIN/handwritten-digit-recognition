use super::state::{ Network, HyperParams, Regularizer };

impl Network {
    fn regularization(Regularizer { l1, l2 }: &Regularizer, value: f64) -> f64 {       
        l1 * value.abs() + l2 * value.powi(2)
    }

    fn compute_costs(&mut self, targets: &[f64]) {
        let output_layer = self.outputs.len() - 1;
        
        for ((cost, output), target) in self.costs[output_layer].iter_mut()
            .zip(self.outputs[output_layer].iter())
            .zip(targets.iter()) 
        {
            *cost = output - target;
        }

        for layer in (0..output_layer).rev() {
            for neuron in 0..self.outputs[layer].len() {               
                self.costs[layer][neuron] = 0.0;

                for (index, weights_prev_layer) in self.weights[layer + 1].iter().enumerate() {
                    self.costs[layer][neuron] += weights_prev_layer[neuron] * self.costs[layer + 1][index];
                }
            }       
        }    
    }

    fn backward_pass(&mut self, layer: usize) {
        let HyperParams { activations, optimizer, regularization, .. } = &self.hyper_params;
        
        for (((((((weights, bias), net_input), cost), moment_1_weights), moment_2_weights), moment_1_bias), moment_2_bias) in self.weights[layer].iter_mut()
            .zip(self.biases[layer].iter_mut())
            .zip(self.net_inputs[layer].iter())
            .zip(self.costs[layer].iter())
            .zip(self.optimizer.moment_1.weights[layer].iter_mut())
            .zip(self.optimizer.moment_2.weights[layer].iter_mut())
            .zip(self.optimizer.moment_1.biases[layer].iter_mut())
            .zip(self.optimizer.moment_2.biases[layer].iter_mut())
        {
            let slope = (activations[layer].derivative)(*net_input);
                       
            for (((weight, prev_layer_output), moment_1_weight), moment_2_weight) in weights.iter_mut()
                .zip(self.outputs[layer - 1].iter())
                .zip(moment_1_weights.iter_mut())
                .zip(moment_2_weights.iter_mut())
            {
                let gradient = 
                    (cost + Self::regularization(&regularization.weights, *weight)) *
                    slope * 
                    prev_layer_output;

                *moment_1_weight = optimizer.beta_1 * *moment_1_weight + (1.0 - optimizer.beta_1) * gradient;
                *moment_2_weight = optimizer.beta_2 * *moment_2_weight + (1.0 - optimizer.beta_2) * gradient.powi(2);

                let corrected_moment_1_weight = *moment_1_weight / (1.0 - optimizer.beta_1.powi(self.optimizer.iteration));
                let corrected_moment_2_weight = *moment_2_weight / (1.0 - optimizer.beta_2.powi(self.optimizer.iteration));

                *weight -= optimizer.alpha * corrected_moment_1_weight / (corrected_moment_2_weight.sqrt() + optimizer.epsilon);
            }

            let gradient = 
                (cost + Self::regularization(&regularization.biases, *bias)) *
                slope;

            *moment_1_bias = optimizer.beta_1 * *moment_1_bias + (1.0 - optimizer.beta_1) * gradient;
            *moment_2_bias = optimizer.beta_2 * *moment_2_bias + (1.0 - optimizer.beta_2) * gradient.powi(2);

            let corrected_moment_1_bias = *moment_1_bias / (1.0 - optimizer.beta_1.powi(self.optimizer.iteration));
            let corrected_moment_2_bias = *moment_2_bias / (1.0 - optimizer.beta_2.powi(self.optimizer.iteration));

            *bias -= optimizer.alpha * corrected_moment_1_bias / (corrected_moment_2_bias.sqrt() + optimizer.epsilon);
        }        
    }
    
    pub fn backward(&mut self, inputs: &[f64], targets: &[f64]) {       
        self.compute_costs(targets);        

        self.backward_pass(self.outputs.len() - 1);
        
        for layer in (1..self.outputs.len() - 1).rev() {
            self.backward_pass(layer);
        }
        
        let HyperParams { activations, optimizer, regularization, .. } = &self.hyper_params;

        for (((((((weights, bias), net_input), cost), moment_1_weights), moment_2_weights), moment_1_bias), moment_2_bias) in self.weights[0].iter_mut()
            .zip(self.biases[0].iter_mut())
            .zip(self.net_inputs[0].iter())
            .zip(self.costs[0].iter())
            .zip(self.optimizer.moment_1.weights[0].iter_mut())
            .zip(self.optimizer.moment_2.weights[0].iter_mut())
            .zip(self.optimizer.moment_1.biases[0].iter_mut())
            .zip(self.optimizer.moment_2.biases[0].iter_mut())
        {
            let slope = (activations[0].derivative)(*net_input);
        
            for (((weight, input), moment_1_weight), moment_2_weight) in weights.iter_mut()
                .zip(inputs)
                .zip(moment_1_weights.iter_mut())
                .zip(moment_2_weights.iter_mut())
            {
                let gradient = 
                    (cost + Self::regularization(&regularization.weights, *weight)) *
                    slope * 
                    input;

                *moment_1_weight = optimizer.beta_1 * *moment_1_weight + (1.0 - optimizer.beta_1) * gradient;
                *moment_2_weight = optimizer.beta_2 * *moment_2_weight + (1.0 - optimizer.beta_2) * gradient.powi(2);

                let corrected_moment_1_weight = *moment_1_weight / (1.0 - optimizer.beta_1.powi(self.optimizer.iteration));
                let corrected_moment_2_weight = *moment_2_weight / (1.0 - optimizer.beta_2.powi(self.optimizer.iteration));

                *weight -= optimizer.alpha * corrected_moment_1_weight / (corrected_moment_2_weight.sqrt() + optimizer.epsilon);
            }          

            let gradient = 
                (cost + Self::regularization(&regularization.biases, *bias)) *
                slope;

            *moment_1_bias = optimizer.beta_1 * *moment_1_bias + (1.0 - optimizer.beta_1) * gradient;
            *moment_2_bias = optimizer.beta_2 * *moment_2_bias + (1.0 - optimizer.beta_2) * gradient.powi(2);

            let corrected_moment_1_bias = *moment_1_bias / (1.0 - optimizer.beta_1.powi(self.optimizer.iteration));
            let corrected_moment_2_bias = *moment_2_bias / (1.0 - optimizer.beta_2.powi(self.optimizer.iteration));

            *bias -= optimizer.alpha * corrected_moment_1_bias / (corrected_moment_2_bias.sqrt() + optimizer.epsilon);
        }
    }
}
