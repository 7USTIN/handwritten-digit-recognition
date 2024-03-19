mod activations;
mod dataset;
mod network;
mod monitor;

use activations::Activation;
use dataset::Dataset;
use network::state::{ Network, HyperParams, Regularization, Regularizer, AdamHyperParams };
use monitor::monitor;

fn main() {
    monitor(|| digit_recognition(), "Handwritten Digit Recognition");
}

fn digit_recognition() {
    let data = monitor(|| Dataset::parse_csv(), "Parsing CSV");

    let hyper_params = HyperParams {
        composition: vec![data.test.inputs[0].len(), 16, 16, data.test.targets[0].len()], 
        activations: Activation::get(&["LEAKY_RELU_001", "LEAKY_RELU_001", "BINARY_STEP"]),
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
    };

    let mut network = monitor(|| Network::new(hyper_params), "Initializing network");

    monitor(|| network.train(&data.train, &data.test, 50), "Training network");
    monitor(|| network.save(), "Saving network hyperparameters");
}
