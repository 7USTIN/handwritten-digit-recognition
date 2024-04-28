use super::{state::{ Network, HyperParams }, optimizations::regularization::Regularization };

impl Network {
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

    fn backward_pass(&mut self, layer: usize, inputs: &[f64]) {
        let HyperParams { activations, learning_rate, optimizer, regularization, .. } = &self.hyper_params;

        let prev_layer_output = match layer == 0 {
            true => inputs,
            false => &self.outputs[layer - 1]
        };
        
        for (((((((((weights, bias), net_input), cost), moment_1_weights), moment_2_weights), moment_1_bias), moment_2_bias), weight_updates), bias_update) in self.weights[layer].iter_mut()
            .zip(self.biases[layer].iter_mut())
            .zip(self.net_inputs[layer].iter())
            .zip(self.costs[layer].iter())
            .zip(self.optimizer.moment_1.weights[layer].iter_mut())
            .zip(self.optimizer.moment_2.weights[layer].iter_mut())
            .zip(self.optimizer.moment_1.biases[layer].iter_mut())
            .zip(self.optimizer.moment_2.biases[layer].iter_mut())
            .zip(self.batch.weight_updates[layer].iter_mut())
            .zip(self.batch.bias_updates[layer].iter_mut())
        {
            let slope = (activations[layer].derivative)(*net_input);
            let current_weight_l2_norm = Regularization::compute_l2_norm(weights);
                       
            for ((((weight, prev_layer_output), moment_1_weight), moment_2_weight), weight_update) in weights.iter_mut()
                .zip(prev_layer_output)
                .zip(moment_1_weights.iter_mut())
                .zip(moment_2_weights.iter_mut())
                .zip(weight_updates.iter_mut())
            {
                let gradient = 
                    (cost + Self::elastic_net_regularization(&regularization.elastic_net.weights, *weight)) *
                    slope *
                    prev_layer_output;


                let (corrected_moment_1, corrected_moment_2) = Self::compute_moments(
                    moment_1_weight, moment_2_weight, optimizer, &gradient, &self.optimizer.iteration
                );

                *weight_update += *weight 
                    - learning_rate.alpha 
                    * corrected_moment_1 
                    / (corrected_moment_2.sqrt() 
                    + optimizer.epsilon);

                if current_weight_l2_norm > regularization.max_norm_constraint {
                    *weight_update *= regularization.max_norm_constraint / current_weight_l2_norm
                }
            }

            let gradient = 
                (cost + Self::elastic_net_regularization(&regularization.elastic_net.biases, *bias)) *
                slope;


            let (corrected_moment_1, corrected_moment_2) = Self::compute_moments(
                moment_1_bias, moment_2_bias, optimizer, &gradient, &self.optimizer.iteration
            );

            *bias_update += *bias 
                - learning_rate.alpha
                * corrected_moment_1
                / (corrected_moment_2.sqrt()
                + optimizer.epsilon);
        }        
    }
    
    pub fn backward(&mut self, inputs: &[f64], targets: &[f64]) {       
        self.compute_costs(targets);        

        for layer in (0..self.outputs.len()).rev() {
            self.backward_pass(layer, inputs);
        }
    }
}
