#![allow(dead_code)]

use crate::{ dataset::Data, network::optimizations::learning_rate::LearningRate };
use super::network::state::{ Network, HyperParams };

use std::time::{ Instant, Duration };
use rand::{ thread_rng, seq::SliceRandom };
use std::cmp::Ordering;

fn print_header(message: &str) {
    println!("\n{:―^50}\n", format!(" {} ", message));    
}

fn print_subheader(message: &str) {    
    println!("\n{:^50}\n", format!("――― {} ―――", message));
}

fn print_centered(message: String) {    
    println!("{:^50}", message);
}

fn print_table(left_col: String, right_col: String) {    
    print!("{:<25}", left_col);
    println!("{:>25}", right_col);
}

fn print_end() {    
    println!("{:―<50}\n", "");
}

pub fn monitor<T, F>(function: F, message: &str) -> T
where
    F: FnOnce() -> T,
{
    print_header(message);
    
    let timestamp = Instant::now();
    let return_value = function();
    
    print_centered(format!("Finished in {:.2?}\n", timestamp.elapsed()));
    print_end();

    return_value
}

pub fn monitor_training(
    epoch: u32, learning_rate: f64, accuracy: f64, cost: f64, duration: Duration, early_stop: bool
) {
    if epoch == 1 {
        print_centered(String::from("Epochs:\n"));        
    }
    
    print_centered(
        format!(
            "[{:0>2?}] LR: {:.2e}, Acc.: {:0>2.2}%, Cost: {:.3}", 
            epoch,
            learning_rate,
            accuracy * 100.0,
            cost
        )
    );

    if early_stop {
        println!();
        print_centered(String::from("Early Stopping"));
        println!();
        print_centered(format!("Avg. Duration: {:.2?}\n", duration / epoch));
    }
}

pub fn statistics(network: &mut Network, data: &Data) {
    let (accuracy, avg_cost) = network.test(data);

    print_header("Neural Network Statistics");

    let HyperParams { 
        composition, regularization, learning_rate, optimizer, batch_size, early_stopping, ..
    } = &network.hyper_params;
    
    let LearningRate { alpha, decay, restart } = learning_rate;

    print_subheader("Composition");

    print_table(
        format!("Input neurons: {}", composition[0]),
        format!("Output neurons: {}", composition.last().unwrap())
    );
    println!("{:<50}\n", format!("Hidden neurons: {:?}", &composition[1..composition.len() - 1]));

    print_table(
        format!("Number of weights: {}", network.weights.iter().map(|layer| layer.iter().map(|neuron| neuron.len()).sum::<usize>()).sum::<usize>()),
        format!("Number of biases: {}", network.biases.iter().map(|layer| layer.len()).sum::<usize>())
    );
    println!();
    
    print_subheader("Regularization");

    print_table(
        format!("L1 Weights: {:e}", regularization.elastic_net.weights.l1),
        format!("L1 Biases: {:e}", regularization.elastic_net.biases.l1)
    );
    print_table(
        format!("L2 Weights: {:e}", regularization.elastic_net.weights.l2),
        format!("L2 Biases: {:e}", regularization.elastic_net.biases.l2)
    );
    println!("{:<50}\n", format!("Max Norm Constraint: {}", regularization.max_norm_constraint));
    println!();

    print_subheader("Adam Optimizer");

    print_table(
        format!("Alpha: {:.2e}", alpha),
        format!("Epsilon: {:e}", optimizer.epsilon)
    );
    print_table(
        format!("Beta 1: {}", optimizer.beta_1),
        format!("Beta 2: {}", optimizer.beta_2)
    );
    println!();

    print_subheader("Training");

    print_table(
        format!("Batch Size: {}", batch_size),
        format!("Iterations: {}", network.optimizer.iteration)
    );
    println!();

    print_subheader("Dropout Rate");
    
    print_table(
        format!("Input Layer: {}%", regularization.dropout_rate.input_layer * 100.0),
        format!("Hidden Layers: {}%", regularization.dropout_rate.hidden_layer * 100.0)
    );
    println!();
    
    print_subheader("Learning Rate");

    if let Some(decay) = &decay {
        print_table(
            format!("Decay Method: {:#?}", decay.method),
            format!("Decay Rate: {:e}", decay.rate)
        );        
    }

    if let Some(restart) = &restart {
        print_table(
            format!("Restart Interval: {}", restart.interval),
            format!("Restart Value: {:e}", restart.alpha)
        );
    }
    println!();
    
    print_subheader("Early Stopping");
    print_table(
        format!("Patience: {}", early_stopping.patience),
        format!("Stability Threshold: {:e}", early_stopping.stability_threshold)
    );
    println!();
    
    print_subheader("Evaluation");

    print_table(
        format!("Accuracy: {:.2}%", accuracy * 100.0),
        format!("Cost: {:.3?}", avg_cost)
    );
    println!();
    
    println!();
    print_end();
}

fn visualize_data(data: &Data, data_index: usize) {
    const SHADES: &str = " .:-=+*#%";
    
    let target = data.targets[data_index]
        .iter()
        .position(|&x| x != 0.0)
        .unwrap_or(0);
    
    print_subheader(&format!("Target: {}", target));

    let sqrt_data_len = f64::sqrt(data.inputs[data_index].len() as f64);

    for (index, intensity) in data.inputs[data_index].iter().enumerate() {
        let scale_intensity = f64::round(*intensity * (SHADES.len() - 1) as f64) as usize; 
        
        for _ in 0..2 {
            print!("{}", SHADES.chars().nth(scale_intensity).unwrap());                
        }

        if (index + 1) as f64 % sqrt_data_len == 0.0 {
            println!();
        }
    }

    println!();
}

fn print_predictions(network: &mut Network, data: &Data, data_index: usize) {
    network.forward(&data.inputs[data_index]);

    print_subheader("Predictions");

    let mut predictions = network.outputs.last().unwrap().iter()
        .enumerate()
        .map(|(number, &output)| (number, output * 100.0))
        .collect::<Vec<(usize, f64)>>();
    
    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    for (number, output) in predictions {            
        match (0.0..10.0).contains(&output) {
            true => print_centered(format!("{}:  {:.2}%", number, output)), // Two spaces 
            false => print_centered(format!("{}: {:.2}%", number, output)) // One space
        } 
    }

    println!();
}

pub fn showcase(network: &mut Network, data: &Data, num_tests: usize) {
    print_header("Showcase");

    let mut rng = thread_rng();
    let data_indices = (0..data.inputs.len()).collect::<Vec<usize>>();
    let random_data_indices = data_indices.choose_multiple(&mut rng, num_tests);

    for &data_index in random_data_indices {
        visualize_data(data, data_index);
        print_predictions(network, data, data_index);
    }
    
    print_end();
}
