mod dataset;
mod network;
mod monitor;

use dataset::Dataset;
use network::{ 
    optimizations::{ 
        activations::{ Activation, ActivationType::* },
        regularization::{ Regularization, ElasticNetRegularization, ElasticNetRegularizer, Dropout },
        learning_rate::{ LearningRate, Restart, Decay, DecayMethod },
        adam::AdamHyperParams,
        early_stopping::EarlyStopping
    }, 
    state::{ Network, HyperParams } 
};
use monitor::{ monitor, statistics, showcase };

fn main() {
    let data = monitor(Dataset::new, "Parsing CSV");

    // Manually tune Hyperparameters
    let hyper_params = HyperParams {
        // Number of neurons in each layer
        composition: vec![data.test.inputs[0].len(), 16, 16, data.test.targets[0].len()],
        // Activation functions for hidden and output layers
        activations: Activation::get(&[LeakyRelu, LeakyRelu, LeakyRelu]),
        regularization: Regularization {
            elastic_net: ElasticNetRegularization {
                weights: ElasticNetRegularizer { l1: 1e-7, l2:  1e-6 },
                biases: ElasticNetRegularizer { l1: 0.0, l2: 0.0 }
            },
            // Probability of a neuron being dropped out
            dropout_rate: Dropout {
                input_layer: 2e-3,
                hidden_layer: 5e-3
            },
            max_norm_constraint: 8.0
        },
        learning_rate: LearningRate {
            alpha: 0.01,
            // Set the learning rate back to a specific value after a specific interval
            restart: Some(Restart {
                interval: 10, // Epochs
                alpha: 1e-3
            }),
            decay: Some(Decay {
                method: DecayMethod::Exponential,
                rate: 0.9,
                // Interval in epochs after which the learning rate is decayed
                step: 2 
            }),
        },
        optimizer: AdamHyperParams {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        },
        batch_size: 4,
        // If the mean difference of the network accuracy over the last 
        // <patience> epochs exceeds <stability_threshold>, stop training
        early_stopping: EarlyStopping {
            stability_threshold: 5e-3,
            patience: 15
        }
    };

    let mut network = monitor(|| Network::new(hyper_params), "Initializing network");
    
    monitor(|| network.train(&data.train, &data.validation), "Training network");
    monitor(|| network.save(), "Saving network parameters");

    // Initalize network with provided Hyperparameters and parameters from the save file
    // let mut network = monitor(|| Network::load(hyper_params), "Loading network parameters");

    statistics(&mut network, &data.test);
    // Visualize handwritten digits and display network predictions
    showcase(&mut network, &data.test, 2);
}
