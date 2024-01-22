use std::time::Instant;

pub fn monitor<T, F>(function: F, message: &str) -> T
where
    F: FnOnce() -> T,
{
    println!();
    println!("-----------------------------");
    println!("{message}...");
    
    let timestamp = Instant::now();
    let return_value = function();
    
    println!();
    println!("Finished in: {:.2?}", timestamp.elapsed());
    println!("-----------------------------");
    println!();

    return_value
}
