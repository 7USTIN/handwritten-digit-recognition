use crate::dataset::Data;
use super::network::state::{ Network, HyperParams };

use std::time::{ Instant, Duration };

fn print_header(message: &str) {
    println!("\n{:―^50}\n", format!(" {} ", message));    
}

fn print_subheader(message: &str) {    
    println!("\n{:^50}\n", format!("――― {} ―――", message));
}

fn print_centered(message: String) {    
    println!("{:^50}\n", message);
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
    
    print_centered(format!("Finished in {:.2?}", timestamp.elapsed()));
    print_end();

    return_value
}

pub fn monitor_training(epochs: u32, epoch: u32, (accuracy, cost): (f64, f64), duration: Duration) {
    print_centered(
        format!(
            "[{:0>2?}] Accuracy: {:.2?}%, Avg. Cost: {:.3?}", 
            epoch + 1,
            accuracy,
            cost
        )
    );

    if epoch + 1 == epochs {
        print_centered(format!("Avg. Duration: {:.2?}", duration / epochs));
    }
}

pub fn statistics(network: &mut Network, testing_data: &Data) {
    let (accuracy, avg_cost) = network.test(testing_data);

    print_header("Neural Network Statistics");

    let HyperParams { composition, regularization, optimizer, batch_size, .. } = &network.hyper_params;

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
        format!("L1 Weights: {:e}", regularization.weights.l1),
        format!("L1 Biases: {:e}", regularization.biases.l1)
    );
    print_table(
        format!("L2 Weights: {:e}", regularization.weights.l2),
        format!("L2 Biases: {:e}", regularization.biases.l2)
    );
    println!();

    print_subheader("Adam Optimizer");

    print_table(
        format!("Alpha: {}", optimizer.alpha),
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
    print_table(
        format!("Accuracy: {}%", accuracy),
        format!("Avg. Cost: {:.3?}", avg_cost)
    );
    println!();

    println!();
    print_end();
}
