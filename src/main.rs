mod activations;
mod dataset;
mod network;
mod monitor;

use dataset::Dataset;
use activations::{ Activation, ActivationType::* };
use network::state::*;
use monitor::{ monitor, statistics, showcase };

fn main() {
    let data = monitor(|| Dataset::parse_csv(), "Parsing CSV");

    const EPOCHS: u32 = 10;

    let hyper_params = HyperParams {
        composition: vec![data.test.inputs[0].len(), 16, 16, data.test.targets[0].len()], 
        activations: Activation::get(&[LeakyRelu, LeakyRelu, LeakyRelu]),
        regularization: Regularization {
            elastic_net: ElasticNetRegularization {
                weights: ElasticNetRegularizer { l1: 1e-7, l2:  1e-6 },
                biases: ElasticNetRegularizer { l1: 0.0, l2: 0.0 }
            },
            dropout_rate: DropoutRate {
                input_layer: 0.0,
                hidden_layer: 0.0
            },
            max_norm_constraint: 8.0
        },
        learning_rate: LearningRate {
            alpha: 0.01,
            restart: Some(LearningRateRestart {
                interval: 10,
                alpha: 0.001
            }),
            decay: Some(LearningRateDecay {
                method: DecayMethod::Exponential,
                rate: 0.9
            }),
        },
        optimizer: AdamHyperParams {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        },
        batch_size: 4,
    };

    let mut network = monitor(|| Network::new(hyper_params), "Initializing network");
    
    monitor(|| network.train(&data.train, &data.test, EPOCHS), "Training network");
    monitor(|| network.save(), "Saving network parameters");

    // let mut network = Network::load(hyper_params);

    statistics(&mut network, &data.test);
    showcase(&mut network, &data.test, 2)
}
