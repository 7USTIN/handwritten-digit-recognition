use rand::{ thread_rng, Rng };

use crate::network::{ state::{ Network, Vec2D }, utils };

pub struct ElasticNetRegularizer {
    pub l1: f64,
    pub l2: f64
}

pub struct ElasticNetRegularization {
    pub weights: ElasticNetRegularizer,
    pub biases: ElasticNetRegularizer    
}

impl Network {
    pub fn elastic_net_regularization(
        ElasticNetRegularizer { l1, l2 }: &ElasticNetRegularizer, value: f64
    ) -> f64 {       
        l1 * value.abs() + l2 * value.powi(2)
    }
}

pub struct Dropout {
    pub input_layer: f64,
    pub hidden_layer: f64,
}

impl Dropout {
    pub fn init_mask(composition: &[usize]) -> Vec2D {
        let mut dropout_mask = utils::zeros_2d_vec(composition, 0);
        
        if let Some(output_layer_mask) = dropout_mask.last_mut() {
            for mask in output_layer_mask.iter_mut() {
                *mask = 1.0;
            }
        }

        dropout_mask
    }
    
    pub fn generate_mask(network: &mut Network) {
        let Regularization { dropout_rate, .. } = &network.hyper_params.regularization;
        
        let mut rng = thread_rng();

        let input_layer_factor = 1.0 / (1.0 - dropout_rate.input_layer);
        let hidden_layer_factor = 1.0 / (1.0 - dropout_rate.hidden_layer);

        for mask in network.dropout_mask[0].iter_mut() {
            *mask = rng.gen_bool(1.0 - dropout_rate.input_layer) as u16 as f64 * input_layer_factor;
        }

        for layer in 1..network.dropout_mask.len() - 1 {
            for neuron in 0..network.dropout_mask[layer].len() {
                network.dropout_mask[layer][neuron] = 
                    rng.gen_bool(1.0 - dropout_rate.hidden_layer) as u16 as f64 * 
                    hidden_layer_factor;
            }
        }
    }

    pub fn set_all_active_mask(network: &mut Network) {
        for layer in 0..network.dropout_mask.len() - 1 {
            for neuron in 0..network.dropout_mask[layer].len() {
                network.dropout_mask[layer][neuron] = 1.0;
            }
        }
    }    

}

pub struct Regularization {
    pub elastic_net: ElasticNetRegularization,
    pub dropout_rate: Dropout,
    pub max_norm_constraint: f64
}

impl Regularization {
    pub fn compute_l2_norm(weights: &[f64]) -> f64 {
        let sum_of_squares = weights.iter().map(|&w| w.powi(2)).sum::<f64>();
        
        sum_of_squares.sqrt()
    }
}
