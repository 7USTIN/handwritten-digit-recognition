use super::{ Network, HyperParams, Regularizer, Vec2D };

impl Network {
    fn regularization(Regularizer { l1, l2 }: &Regularizer, value: f64) -> f64 {       
        l1 * value.abs() + l2 * value.powi(2)
    }

    fn compute_costs(&mut self, costs: &mut Vec2D, targets: &[f64]) {
        let output_layer = self.outputs.len() - 1;
        
        for ((cost, output), target) in costs[output_layer].iter_mut()
            .zip(self.outputs[output_layer].iter())
            .zip(targets.iter()) 
        {
            *cost = output - target;
        }

        for layer in (0..output_layer).rev() {
            for neuron in 0..self.outputs[layer].len() {               
                costs[layer][neuron] = 0.0;

                for (index, weights_prev_layer) in self.weights[layer + 1].iter().enumerate() {
                    costs[layer][neuron] += weights_prev_layer[neuron] * costs[layer + 1][index];
                }
            }       
        }    
    }

    fn backward_pass(&mut self, layer: usize, costs: &mut Vec2D) {
        let HyperParams { activations, learning_rate, regularization, .. } = &self.hyper_params;
        
        for (((weights, bias), net_input), cost) in self.weights[layer].iter_mut()
            .zip(self.biases[layer].iter_mut())
            .zip(self.net_inputs[layer].iter())
            .zip(costs[layer].iter())
        {
            let slope = (activations[layer].derivative)(*net_input);
            
            for (weight, prev_layer_output) in weights.iter_mut().zip(self.outputs[layer - 1].iter()) {
                *weight -= 
                    learning_rate * 
                    (cost + Self::regularization(&regularization.weight, *weight)) *
                    slope * 
                    prev_layer_output;
            }

            *bias -= learning_rate * (cost + Self::regularization(&regularization.bias, *bias)) * slope;
        }        
    }
    
    pub fn backward(&mut self, inputs: &[f64], targets: &[f64], costs: &mut Vec2D) {       
        self.compute_costs(costs, targets);        

        self.backward_pass(self.outputs.len() - 1, costs);
        
        for layer in (1..self.outputs.len() - 1).rev() {
            self.backward_pass(layer, costs);
        }
        
        let HyperParams { activations, learning_rate, regularization, .. } = &self.hyper_params;

        for (((weights, bias), net_input), cost) in self.weights[0].iter_mut()
            .zip(self.biases[0].iter_mut())
            .zip(self.net_inputs[0].iter())
            .zip(costs[0].iter())
        {
            let slope = (activations[0].derivative)(*net_input);
        
            for (weight, input) in weights.iter_mut().zip(inputs) {
                *weight -= learning_rate * (cost + Self::regularization(&regularization.weight, *weight)) * slope * input;
            }
           
            *bias -= learning_rate * (cost + Self::regularization(&regularization.bias, *bias)) * slope;
        }
    }
}
