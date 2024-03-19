use std::time::Instant;

pub fn monitor<T, F>(function: F, message: &str) -> T
where
    F: FnOnce() -> T,
{
    println!("\n-----------------------------");
    println!("{message}...");
    
    let timestamp = Instant::now();
    let return_value = function();
    
    println!("\nFinished in: {:.2?}", timestamp.elapsed());
    println!("-----------------------------\n");

    return_value
}

pub fn monitor_training(epoch: u32, accuracy: f64) {
    println!(
        "[{:0>2?}] Accuracy: {:.2?}%", 
        epoch + 1,
        accuracy
    );
}
