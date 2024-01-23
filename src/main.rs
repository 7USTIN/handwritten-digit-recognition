mod activations;
mod dataset;
mod network;
mod monitor;

use activations::Activation;
use dataset::Dataset;
use network::{ Network, HyperParams, Regularization, Regularizer };
use monitor::monitor;

fn digit_recognition() {
    let data = monitor(|| Dataset::parse_csv(), "Parsing CSV");

    let hyper_params = HyperParams {
        composition: vec![data.test.inputs[0].len(), 16, 16, data.test.targets[0].len()], 
        activations: Activation::get(&["LEAKY_RELU_001", "LEAKY_RELU_001", "BINARY_STEP"]),
        learning_rate: 0.01,
        regularization: Regularization {
            weight: Regularizer { l1: 0.0, l2: 0.0 },
            bias: Regularizer { l1: 0.0, l2: 0.0 }
        }
    };

    let mut network = monitor(|| Network::new(hyper_params), "Initializing network");

    monitor(|| network.train(&data.train, &data.test, 10), "Training network");
    monitor(|| network.save(), "Saving network hyperparameters");
}

fn main() {
    monitor(|| digit_recognition(), "Handwritten Digit Recognition");
}
