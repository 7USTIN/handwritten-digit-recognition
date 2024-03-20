use std::time::{ Instant, Duration };

pub fn monitor<T, F>(function: F, message: &str) -> T
where
    F: FnOnce() -> T,
{
    println!("\n{:―^50}\n", format!(" {} ", message));
    
    let timestamp = Instant::now();
    let return_value = function();
    
    println!("{:^50}\n", format!("Finished in {:.2?}", timestamp.elapsed()));
    println!("{:―<50}\n", "");

    return_value
}

pub fn monitor_training(epochs: u32, epoch: u32, (accuracy, cost): (f64, f64), duration: Duration) {
    println!(
        "{:^50}",
        format!(
            "[{:0>2?}] Accuracy: {:.2?}%, Avg. Cost: {:.3?}", 
            epoch + 1,
            accuracy,
            cost
        ),
    );

    if epoch + 1 == epochs {
      println!("\n{:^50}\n", format!("Avg. Duration: {:.2?}", duration / epochs));  
    }
}
