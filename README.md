# RL-for-scheduling-power-allocation

## Train
Run train.py to train the DQN agents and models are saved in ./simulations folder
## Check convergence (optional)
Run check_convergence.py to get the learning curve and neuron coefficients change in training.
![Figure 2022-09-10 165958 (0)](https://user-images.githubusercontent.com/91915172/191305536-bcde456e-8eaf-483e-9ed3-570fa57d452c.png)
![Figure 2022-09-10 165958 (2)](https://user-images.githubusercontent.com/91915172/191305729-232a3b21-ce29-4048-8739-ec6930ceace4.png)
## Test
Run test_delay_models.py to test multiple models under the same environment. (Optional)
Run test_1_model.py to test 1 model in a continously changing environment. (Optional)
Run test_compare_2_benchmark.py to compare the trained model with benchmark algorithms including FP(fractional programming) and wmmse.
## Generate figures
Run generate_result_for_models.py to generate result figures like the following.
![Figure 2022-09-10 182517 (1)](https://user-images.githubusercontent.com/91915172/191302721-c6a09a34-4581-4e1d-8b7c-3172ad779151.png)
![Figure 2022-09-11 125856 (1)](https://user-images.githubusercontent.com/91915172/191302935-f6d2c043-69f9-44dd-9bd1-39bdaf874a58.png)

## Agents and Envrionements
Agents are defined in DQN.py while environemts are defined in env.py

