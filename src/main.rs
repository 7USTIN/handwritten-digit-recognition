mod activations;
mod dataset;
mod network;

use activations::Activation;
use dataset::Dataset;
use network::{ Network, HyperParams, Regularization, Regularizer };

fn main() {
    let data = Dataset::parse_csv();

    let hyper_params = HyperParams {
        composition: vec![data.test.inputs[0].len(), 16, 16, data.test.targets[0].len()], 
        activations: Activation::get(&["LEAKY_RELU_001", "LEAKY_RELU_001", "BINARY_STEP"]),
        learning_rate: 0.01,
        regularization: Regularization {
            weight: Regularizer { l1: 0.0001, l2: 0.0001 },
            bias: Regularizer { l1: 0.000_000_1, l2: 0.000_000_1 }
        }
    };

    let mut network = Network::new(hyper_params);

    network.train(&data.train, 10);
    network.test(&data.test);
    network.save();
}
