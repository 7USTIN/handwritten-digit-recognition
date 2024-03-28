#![allow(dead_code)]

mod activations;
mod dataset;
mod network;
mod monitor;

use activations::Activation;
use dataset::Dataset;
use network::state::{ Network, HyperParams, Regularization, Regularizer, AdamHyperParams };
use monitor::{ monitor, statistics, showcase };

fn main() {
    let data = monitor(|| Dataset::parse_csv(), "Parsing CSV");

    const EPOCHS: u32 = 10;

    let hyper_params = HyperParams {
        composition: vec![data.test.inputs[0].len(), 16, 16, data.test.targets[0].len()], 
        activations: Activation::get(&["LEAKY_RELU_001", "LEAKY_RELU_001", "LEAKY_RELU_001"]),
        regularization: Regularization {
            weights: Regularizer { l1: 1e-7, l2:  1e-6},
            biases: Regularizer { l1: 1e-9, l2: 1e-8 }
        },
        optimizer: AdamHyperParams {
            alpha: 0.005,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        },
        batch_size: 4,
    };

    // let mut network = monitor(|| Network::new(hyper_params), "Initializing network");

    // monitor(|| network.train(&data.train, &data.test, EPOCHS), "Training network");
    // monitor(|| network.save(), "Saving network parameters");

    let mut network = Network::load(hyper_params);

    statistics(&mut network, &data.test);
    showcase(&mut network, &data.test, 2)
}
