This repo contains code for optimization algorithms for gradient descent , backpropagation through time , and a hybrid backpropagation through time and real time recurrent learning algorithm for the lstm model.

To use just import the desired module.
```python
from rnn import RNN
model = RNN()
```

Some results:

# 1. learning mean function with delta rule
<img width="576" height="432" alt="image" src="https://github.com/user-attachments/assets/2330c08a-e3bc-4e3b-b50d-a873722ef42b" />

# 2. learning sliding window mean with backpropagation through time

<img width="593" height="432" alt="image" src="https://github.com/user-attachments/assets/310ab0ff-5229-4a61-b697-8d8a34ee61f9" />

# 3. learning sliding window mean with real time recurrent learning 

<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/777bb1ee-c8b6-41a5-a3b2-976e195c925e" />

# 4. hybrid backpropagation through time and real time recurrent learning algorithm for the lstm model on california housing dataset
<img width="576" height="432" alt="image" src="https://github.com/user-attachments/assets/94f81c5e-7bb9-42a1-b204-cdeda4e03764" />


