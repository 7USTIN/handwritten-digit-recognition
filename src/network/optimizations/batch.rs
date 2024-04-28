use crate::network::{ state::{ Vec2D, Network }, utils };

pub struct Batch {
    pub weight_updates: Vec<Vec2D>,
    pub bias_updates: Vec2D
}

impl Batch {
    pub fn new(composition: &[usize]) -> Self {
        Self {
            weight_updates: utils::zeros_3d_vec(composition),
            bias_updates: utils::zeros_2d_vec(composition, 1)
        }
    }
    
    pub fn update(network: &mut Network, chunk_size: f64) {
        for (weights, weight_updates) in network.weights.iter_mut()
            .zip(network.batch.weight_updates.iter_mut())
        {
            for (weights, weight_updates) in weights.iter_mut()
                .zip(weight_updates.iter_mut())
            {
                for (weight, weight_update) in weights.iter_mut()
                    .zip(weight_updates.iter_mut())
                {
                    *weight = *weight_update / chunk_size;
                    *weight_update = 0.0;
                }
            }
        }
        
        for (biases, bias_updates) in network.biases.iter_mut()
            .zip(network.batch.bias_updates.iter_mut())
        {
            for (bias, bias_update) in biases.iter_mut()
                .zip(bias_updates.iter_mut()) 
            {
                *bias = *bias_update / chunk_size;
                *bias_update = 0.0;
            }
        }
    }
}
