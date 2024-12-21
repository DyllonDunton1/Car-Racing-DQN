# Car-Racing-DQN
This is a project which comes from my desire to learn reinforcement learning. I started off by doing a project in my ECE591 Deep learning course where I tried to use tensorflow to implement q learning in the openai-gym 'Car-Racing-v3' environment. This did not work out well, but recently (November 2024) I have come back to this project with a new understanding. I began by attempting to implement Q-learning with the much simpler cart-pole environment.

# Cart Pole
![cartpole](https://github.com/user-attachments/assets/7762eca4-1209-47bc-8793-aa8ab4c8fbff)


I was able to use Q-Learning to train a simple fully connected model with pytorch to balance the pole on top of the cart while staying in the center of the screen. It is so good that I have not been able to measure the point in which it fails since it runs forever.

# Car Racing Q-Learning
![carracing](https://github.com/user-attachments/assets/9f186281-e001-4f6e-99db-00228a8f46dd)


I am currently working on this now, and have not had any promising results, but that is simply due to the need for a long training time. The next step will be working to preprocess the image from the car-racing to simplify it before sending to the CNN
