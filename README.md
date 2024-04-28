# Handwritten Digit Recognition
***Written in pure Rust from Scratch***

<br>

The project was developed by me as a hobby project to learn Rust and to understand neural networks.

The neural network is trained on the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database).

<br>

## Neural Network Features

- [ ] Performance: Utilize Compute Shaders using [Rust wgpu](https://github.com/gfx-rs/wgpu)
- [ ] Automatic Hyperparameter Optimization
- [X] Adam Optimizer
- [X] Batch Training
- [X] Variety of Activation Funictions
- [ ] Regularization
    - [X] Elastic Net Regularization
    - [X] Dropout
    - [X] Max-Norm Constraint
    - [ ] Layer Normalization
- [X] Early Stopping
- [X] Learning Rate Decay and Restart
- [ ] Saving and Loading Parameters (currently  bugged)

<br>

## Example Prediction
```
                ――― Target: 5 ―――                 

                                                        
                                                        
                                                        
                                                        
                                  ..==**%%**            
                            ..==##%%%%%%%%%%::          
                      ::==++%%%%%%%%%%%%%%%%::          
                    ==%%%%%%%%%%%%%%##++::..            
                    **%%%%%%%%##==..                    
                    ++%%%%++--                          
                    ##%%##                              
                  --%%%%##                              
                  ++%%%%##..::----==::                  
                ..##%%%%%%%%%%%%%%%%%%--                
                **%%%%%%%%%%%%%%##**####..              
                ##%%%%%%%%**==..    ::%%==              
                ==%%%%%%--          ::%%++              
                  ::--..            ==%%++              
                                  ::%%%%++              
                  ::..            ++%%%%--              
              ..##%%**........==**%%%%##                
              ::%%%%%%%%%%%%%%%%%%%%%%==                
              ..##%%%%%%%%%%%%%%%%##==                  
                ..--++**%%%%%%##==..                    
                                                        
                                                        
                                                        
                                                        


               ――― Predictions ―――                

                    5: 111.07%                    
                    6: -0.02%                     
                    9: -0.07%                     
                    1: -0.15%                     
                    8: -0.19%                     
                    0: -0.46%                     
                    7: -0.48%                     
                    4: -0.51%                     
                    3: -0.53%                     
                    2: -0.64%   
```

<br>

## Setup

1. Clone the GitHub repository.
    ```
    git clone https://github.com/7USTIN/handwritten-digit-recognition.git
    ```

2. Create a directory named "dataset" in the root directory of the project.

3. Download the "[mnist_train.csv](https://drive.google.com/file/d/1eEKzfmEu6WKdRlohBQiqi3PhW_uIVJVP/view)" and "[mnist_test.csv](https://drive.google.com/file/d/1eEKzfmEu6WKdRlohBQiqi3PhW_uIVJVP/view)" files and move them to "/dataset".

<br>

## Example Output

```
$ cargo run --release

―――――――――――――――――― Parsing CSV ―――――――――――――――――――

                Finished in 2.25s
                
――――――――――――――――――――――――――――――――――――――――――――――――――


―――――――――――――― Initializing network ――――――――――――――

              Finished in 732.87µs
               
――――――――――――――――――――――――――――――――――――――――――――――――――


―――――――――――――――― Training network ――――――――――――――――

                     Epochs:
                     
   [01] LR: 1.00e-2, Acc.: 90.42%, Cost: 0.106    
   [02] LR: 1.00e-2, Acc.: 91.59%, Cost: 0.093    
   [03] LR: 8.10e-3, Acc.: 92.23%, Cost: 0.088    
   [04] LR: 8.10e-3, Acc.: 93.14%, Cost: 0.082    
   [05] LR: 5.31e-3, Acc.: 93.83%, Cost: 0.080    
   [06] LR: 5.31e-3, Acc.: 93.88%, Cost: 0.087    
   [07] LR: 2.82e-3, Acc.: 94.21%, Cost: 0.073    
   [08] LR: 2.82e-3, Acc.: 94.15%, Cost: 0.070    
   [09] LR: 1.22e-3, Acc.: 94.58%, Cost: 0.065    
   [10] LR: 1.22e-3, Acc.: 94.73%, Cost: 0.064    
   [11] LR: 1.00e-3, Acc.: 94.81%, Cost: 0.061    
   [12] LR: 1.00e-3, Acc.: 94.78%, Cost: 0.061    
   [13] LR: 8.10e-4, Acc.: 94.96%, Cost: 0.060    
   [14] LR: 8.10e-4, Acc.: 94.88%, Cost: 0.060    
   [15] LR: 5.31e-4, Acc.: 95.16%, Cost: 0.059    

                 Early Stopping                  

              Avg. Duration: 4.09s
               
               Finished in 63.39s
                
――――――――――――――――――――――――――――――――――――――――――――――――――


―――――――――――――――――――― Showcase ――――――――――――――――――――


                ――― Target: 3 ―――                 

                                                        
                                                        
                                                        
                                                        
                                                        
                --++%%%%%%%%%%%%##..                    
                ++%%%%%%%%%%%%%%%%%%--                  
                --%%%%%%%%%%%%%%%%%%%%##                
                ..%%%%%%%%####%%%%%%%%%%--              
                  **##==--..  ::**%%%%%%##..            
                              ::##%%%%%%%%::            
                  ..      ::++##%%%%%%%%##              
                  ++++++**%%%%%%%%%%%%##::              
                  ++%%%%%%%%%%%%%%%%**                  
                  ++%%%%%%%%%%%%%%%%**                  
                  ==%%%%%%##----%%%%%%==                
                    ====::..    --%%%%##..              
              ::              ..==%%%%%%::              
            ::%%==            ==%%%%%%%%..              
            ==%%%%::    ..--**%%%%%%%%%%..              
            %%%%%%########%%%%%%%%%%%%##                
            %%%%%%%%%%%%%%%%%%%%%%%%==                  
            **%%%%%%%%%%%%%%%%%%%%==                    
            --%%%%%%%%%%%%%%%%==                        
              ==**%%%%++==..                            
                                                        
                                                        
                                                        


               ――― Predictions ―――                

                    3: 111.80%                    
                    8: 12.93%                     
                    9: 10.79%                     
                    5:  0.90%                     
                    4: -0.07%                     
                    1: -0.14%                     
                    7: -0.21%                     
                    6: -0.33%                     
                    0: -0.43%                     
                    2: -0.50%                    


                 ――― Target: 6 ―――                 

                                                        
                                                        
                                                        
                          ==**::                        
                        ++%%%%##                        
                      ++%%%%%%--                        
                    ++%%%%%%++                          
                    **%%%%##..                          
                  --##%%%%::                            
                ..**%%%%##..                            
                ::%%%%%%::                              
                ::%%%%++    ::--------                  
                ::%%%%++  ::%%%%%%%%%%**                
                ::%%%%++::%%%%%%%%%%%%%%::              
                ==%%%%####%%%%%%++**%%%%##..            
                --%%%%%%%%%%##--..  ++%%%%::            
                ::%%%%%%%%++..      ++%%%%--            
                  ++%%%%%%..      ::##%%%%::            
                    **%%%%**::::==##%%%%##..            
                    ::##%%%%%%%%%%%%%%%%==              
                      ++%%%%%%%%%%%%##==                
                        ++%%%%%%%%++..                  
                        ::%%**==                        
                                                        
                                                        
                                                        
                                                        
                                                        


               ――― Predictions ―――                

                    6: 97.65%                     
                    5: -0.07%                     
                    9: -0.09%                     
                    8: -0.17%                     
                    2: -0.34%                     
                    0: -0.41%                     
                    4: -0.42%                     
                    1: -0.51%                     
                    7: -0.52%                     
                    3: -0.59%                       


――――――――――――――――――――――――――――――――――――――――――――――――――


――――――――――― Neural Network Statistics ――――――――――――


               ――― Composition ―――                

Input neurons: 784              Output neurons: 10
Hidden neurons: [16, 16]                          

Number of weights: 12960      Number of biases: 42


              ――― Regularization ―――              

L1 Weights: 1e-7                    L1 Biases: 0e0
L2 Weights: 1e-6                    L2 Biases: 0e0
Max Norm Constraint: 8                            



              ――― Adam Optimizer ―――              

Alpha: 5.31e-4                       Epsilon: 1e-8
Beta 1: 0.9                          Beta 2: 0.999


                 ――― Training ―――                 

Batch Size: 4                   Iterations: 720000


               ――― Dropout Rate ―――               

Input Layer: 0.2%              Hidden Layers: 0.5%


              ――― Learning Rate ―――               

Decay Method: Exponential         Decay Rate: 9e-1
Restart Interval: 10           Restart Value: 1e-3


              ――― Early Stopping ―――              

Patience: 15             Stability Threshold: 5e-3


                ――― Evaluation ―――                

Accuracy: 95.05%                       Cost: 0.068


――――――――――――――――――――――――――――――――――――――――――――――――――
```
