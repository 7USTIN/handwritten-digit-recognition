mod activations;
mod dataset;
mod network;

use activations::Activation;
use dataset::Dataset;
use network::{ Network, HyperParams, Regularization, Regularizer };

fn main() {
    let data = Dataset::parse_csv();

    let hyper_params = HyperParams {
        composition: vec![data.test.inputs[0].len(), 4, data.test.targets[0].len()], 
        activations: Activation::get(&["LEAKY_RELU_001", "BINARY_STEP"]),
        learning_rate: 0.01,
        regularization: Regularization {
            weight: Regularizer { l1: 0.0001, l2: 0.0001 },
            bias: Regularizer { l1: 0.0000001, l2: 0.0000001 }
        }
    };

    let mut network = Network::new(hyper_params);

    network.train(data.train, 1);
    network.test(data.test);
    network.save();
}
